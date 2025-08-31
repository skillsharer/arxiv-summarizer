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
import json
import asyncio
from tweet_variations import TweetVariations

load_dotenv()

OPENAI_BEARER_TOKEN = os.getenv('OPENAI_BEARER_TOKEN')
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Twitter authentication method configuration
TWITTER_AUTH_METHOD = os.getenv("TWITTER_AUTH_METHOD", "tweepy").lower()  # tweepy or twikit
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")  # For twikit method
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")  # For twikit method
TWITTER_EMAIL = os.getenv("TWITTER_EMAIL")  # For twikit method (optional)

# Global variables for Twitter clients
twitter_client = None
twitter_api = None
twikit_client = None

async def setup_twikit_client(force_reauth=False):
    """Setup and authenticate twikit client with session persistence and validation"""
    global twikit_client
    
    try:
        from twikit import Client
        
        twikit_client = Client('en-US')
        cookies_file = 'twitter_cookies.json'
        
        # Try to load existing cookies unless forced to re-authenticate
        if not force_reauth and os.path.exists(cookies_file):
            try:
                twikit_client.load_cookies(cookies_file)
                print("🍪 Loaded existing Twitter session cookies")
                
                # Validate the session by trying a simple operation
                print("🔍 Validating session...")
                try:
                    # Test if we're properly authenticated
                    await twikit_client.get_user_by_screen_name(TWITTER_USERNAME)
                    print("✅ Session validation successful")
                    return True
                except Exception as e:
                    print(f"⚠️  Session validation failed: {e}")
                    print("🔄 Session expired, will re-authenticate...")
                    # Remove invalid cookies
                    try:
                        os.remove(cookies_file)
                    except:
                        pass
                    
            except Exception as e:
                print(f"⚠️  Failed to load cookies: {e}")
        
        # Fresh login required
        if TWITTER_USERNAME and TWITTER_PASSWORD:
            print("🔐 Performing fresh login to Twitter with twikit...")
            try:
                await twikit_client.login(
                    auth_info_1=TWITTER_USERNAME,
                    auth_info_2=TWITTER_EMAIL or TWITTER_USERNAME,
                    password=TWITTER_PASSWORD
                )
                
                # Save fresh cookies
                twikit_client.save_cookies(cookies_file)
                print("✅ Successfully authenticated with twikit and saved fresh session")
                return True
            except Exception as login_error:
                print(f"❌ Login failed: {login_error}")
                # Clean up any partial session files
                try:
                    if os.path.exists(cookies_file):
                        os.remove(cookies_file)
                except:
                    pass
                return False
        else:
            print("❌ Missing Twitter credentials for twikit method")
            return False
            
    except Exception as e:
        print(f"❌ Twikit authentication setup failed: {e}")
        return False

def clear_twitter_session():
    """Clear stored Twitter session cookies to force fresh authentication"""
    cookies_file = 'twitter_cookies.json'
    try:
        if os.path.exists(cookies_file):
            os.remove(cookies_file)
            print("🧹 Cleared Twitter session cookies")
            return True
        else:
            print("ℹ️  No session cookies found to clear")
            return False
    except Exception as e:
        print(f"⚠️  Failed to clear session cookies: {e}")
        return False

def handle_twitter_auth_failure():
    """Handle Twitter authentication failures with helpful suggestions"""
    print("\n" + "="*60)
    print("🚨 TWITTER AUTHENTICATION ISSUE DETECTED")
    print("="*60)
    print("Twitter has temporarily restricted login due to 'unusual activity'.")
    print("This is common when using automation tools.")
    print("\n📋 SOLUTIONS (try in order):")
    print("1. ⏰ WAIT 15-30 minutes and try again")
    print("2. 🌐 Log into Twitter manually in browser first")
    print("3. 🔄 Switch to tweepy method temporarily:")
    print("   Add to .env: TWITTER_AUTH_METHOD=tweepy")
    print("4. 🧹 Clear session and retry:")
    print("   python -c \"import main; main.clear_twitter_session()\"")
    print("5. 🐛 Test with debug mode:")
    print("   python -c \"import main; main.tweet_arxiv_papers(debug=True)\"")
    print("\n💡 TIP: Tweepy method is more stable for production use!")
    print("="*60 + "\n")

def setup_tweepy_client():
    """Setup traditional tweepy client"""
    global twitter_client, twitter_api
    
    try:
        twitter_client = tweepy.Client(
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
        twitter_api = tweepy.API(auth)
        print("✅ Successfully authenticated with tweepy")
        return True
    except Exception as e:
        print(f"❌ Tweepy authentication failed: {e}")
        return False

async def send_tweet_twikit(text, media_paths=None, reply_to_id=None, retry_auth=True):
    """Send tweet using twikit method with better rate limits and auth retry"""
    global twikit_client
    
    try:
        if not twikit_client:
            success = await setup_twikit_client()
            if not success:
                return None
        
        # Upload media if provided
        media_ids = []
        if media_paths:
            for media_path in media_paths:
                if os.path.exists(media_path):
                    media = await twikit_client.upload_media(media_path)
                    media_ids.append(media)
        
        # Send tweet with reply if specified
        if reply_to_id:
            if media_ids:
                tweet = await twikit_client.create_tweet(text=text, media_ids=media_ids, reply_to=reply_to_id)
            else:
                tweet = await twikit_client.create_tweet(text=text, reply_to=reply_to_id)
        else:
            if media_ids:
                tweet = await twikit_client.create_tweet(text=text, media_ids=media_ids)
            else:
                tweet = await twikit_client.create_tweet(text=text)
            
        print(f"📤 Tweet sent successfully via twikit (ID: {tweet.id})")
        return tweet.id
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for authentication errors
        if ("401" in error_msg or "Could not authenticate" in error_msg or "code\":32" in error_msg) and retry_auth:
            print(f"🔄 Authentication error detected, attempting fresh login...")
            try:
                # Force re-authentication
                success = await setup_twikit_client(force_reauth=True)
                if success:
                    print("🔄 Retrying tweet with fresh authentication...")
                    # Retry once with fresh auth (but don't retry again to avoid infinite loop)
                    return await send_tweet_twikit(text, media_paths, reply_to_id, retry_auth=False)
                else:
                    print("❌ Re-authentication failed")
                    handle_twitter_auth_failure()
                    return None
            except Exception as retry_error:
                print(f"❌ Re-authentication attempt failed: {retry_error}")
                if "unusual" in str(retry_error).lower() or "tiltás" in str(retry_error).lower():
                    handle_twitter_auth_failure()
                return None
        elif "duplicate" in error_msg.lower() or "187" in error_msg:
            print(f"⚠️  Duplicate tweet detected: {error_msg}")
            print("💡 Tip: Try running again or modify the tweet content to make it unique")
        elif "event loop" in error_msg.lower():
            print(f"⚠️  Event loop error: {error_msg}")
            print("💡 Tip: This is usually a temporary issue, try running again")
        else:
            print(f"❌ Failed to send tweet via twikit: {e}")
        return None

def send_tweet_tweepy(text, media_paths=None, reply_to_id=None):
    """Send tweet using traditional tweepy method"""
    global twitter_client, twitter_api
    
    try:
        if not twitter_client:
            success = setup_tweepy_client()
            if not success:
                return None
        
        # Upload media if provided
        media_ids = []
        if media_paths:
            for media_path in media_paths:
                if os.path.exists(media_path):
                    media = twitter_api.media_upload(media_path)
                    media_ids.append(media.media_id)
        
        # Send tweet with reply if specified
        if reply_to_id:
            if media_ids:
                tweet = twitter_client.create_tweet(text=text, media_ids=media_ids, in_reply_to_tweet_id=reply_to_id)
            else:
                tweet = twitter_client.create_tweet(text=text, in_reply_to_tweet_id=reply_to_id)
        else:
            if media_ids:
                tweet = twitter_client.create_tweet(text=text, media_ids=media_ids)
            else:
                tweet = twitter_client.create_tweet(text=text)
            
        print(f"📤 Tweet sent successfully via tweepy (ID: {tweet.data['id']})")
        return tweet.data['id']
        
    except Exception as e:
        error_msg = str(e)
        if "duplicate" in error_msg.lower() or "187" in error_msg:
            print(f"⚠️  Duplicate tweet detected: {error_msg}")
            print("💡 Tip: Try running again or modify the tweet content to make it unique")
        else:
            print(f"❌ Failed to send tweet via tweepy: {e}")
        return None

async def send_tweet_unified(text, media_paths=None):
    """Unified tweet sending function that handles both authentication methods"""
    if TWITTER_AUTH_METHOD == "twikit":
        return await send_tweet_twikit(text, media_paths)
    else:
        return send_tweet_tweepy(text, media_paths)

def send_tweet_sync(text, media_paths=None, reply_to_id=None):
    """Synchronous wrapper for tweet sending with reply support"""
    if TWITTER_AUTH_METHOD == "twikit":
        # Handle async function with thread-based approach
        import threading
        import concurrent.futures
        
        def run_async_in_thread():
            """Run async function in a separate thread with its own event loop"""
            return asyncio.run(send_tweet_twikit(text, media_paths, reply_to_id))
        
        try:
            # Always use thread-based approach to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_in_thread)
                return future.result(timeout=60)  # 60 second timeout
        except Exception as e:
            print(f"❌ Failed to send tweet via twikit (thread error): {e}")
            return None
    else:
        return send_tweet_tweepy(text, media_paths, reply_to_id)

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
        
        if not name_parts:  # Handle empty name case
            return []
            
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
            results = data.get('result', [])
            if results:  # Check if results exist and is not None
                for result in results:
                    orcid_identifier = result.get('orcid-identifier')
                    if orcid_identifier:  # Check if orcid-identifier exists
                        orcid_id = orcid_identifier.get('path')
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
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 3, 5, 9 seconds
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

def format_tweet_with_author_tags(base_text, authors, author_handles, max_length=4000):
    """
    Format tweet text with author tags, ensuring we stay within Twitter's character limit.
    If no social media handles are found, mentions authors by name.
    Note: Twitter now allows up to 4,000 characters for tweets.
    """
    # Start with the base text
    tweet_text = base_text
    
    # Collect handles to tag and authors without handles
    handles_to_tag = []
    authors_without_handles = []
    
    for author in authors:
        author_name = str(author.name).strip()
        if author_name in author_handles:
            handles_to_tag.append(f"@{author_handles[author_name]}")
        else:
            authors_without_handles.append(author_name)
    
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
    
    # If no social media handles found, mention authors by name (but keep it concise)
    elif authors_without_handles:
        # Only mention first 3 authors to avoid making tweet too long
        authors_to_mention = authors_without_handles[:3]
        if len(authors_without_handles) > 3:
            author_names_string = ", ".join(authors_to_mention) + " et al."
        else:
            author_names_string = ", ".join(authors_to_mention)
        
        author_tag_string = f" Authors: {author_names_string}"
        
        # Check if adding author names would exceed character limit
        if len(tweet_text) + len(author_tag_string) <= max_length:
            tweet_text += author_tag_string
        else:
            # Try with just first author
            single_author_string = f" Author: {authors_to_mention[0]}"
            if len(tweet_text) + len(single_author_string) <= max_length:
                tweet_text += single_author_string
    
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

def is_quality_image(image_path, min_size=(200, 200), min_colors=20, max_text_ratio=0.8):
    """
    Determines if an image is suitable for inclusion in a GIF based on quality metrics.
    
    Args:
        image_path (str): Path to the image file
        min_size (tuple): Minimum width and height in pixels
        min_colors (int): Minimum number of unique colors required
        max_text_ratio (float): Maximum ratio of potential text pixels (0.0-1.0)
    
    Returns:
        bool: True if image passes quality checks, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            
            # 1. Size filter - reject tiny images (likely icons, emojis, or symbols)
            if width < min_size[0] or height < min_size[1]:
                print(f"❌ Rejected {os.path.basename(image_path)}: Too small ({width}x{height})")
                return False
            
            # 2. Aspect ratio filter - reject extremely narrow or wide images
            aspect_ratio = width / height
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                print(f"❌ Rejected {os.path.basename(image_path)}: Bad aspect ratio ({aspect_ratio:.2f})")
                return False
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # 3. Color diversity filter - reject images with too few colors (likely simple graphics/text)
            # Reshape to 2D array of RGB values and count unique colors more efficiently
            pixels = img_array.reshape(-1, 3)
            # Convert to tuples for proper hashing
            unique_colors = len(set(map(tuple, pixels)))
            
            if unique_colors < min_colors:
                print(f"❌ Rejected {os.path.basename(image_path)}: Too few colors ({unique_colors})")
                return False
            
            # 4. Monochrome/blank image filter
            # Check for mostly black, white, or single-color images
            # Be more lenient for research figures which often have white backgrounds
            total_pixels = width * height
            
            # Count black pixels
            black_pixels = np.sum(np.all(img_array == [0, 0, 0], axis=2))
            black_ratio = black_pixels / total_pixels
            
            # Count white pixels
            white_pixels = np.sum(np.all(img_array == [255, 255, 255], axis=2))
            white_ratio = white_pixels / total_pixels
            
            # Count near-white pixels (for PDFs with slight background color variations)
            near_white_pixels = np.sum(np.all(img_array > [240, 240, 240], axis=2))
            near_white_ratio = near_white_pixels / total_pixels
            
            # More lenient thresholds for research figures (they often have white backgrounds)
            if black_ratio > 0.9 or (white_ratio > 0.95 and unique_colors < 10):
                print(f"❌ Rejected {os.path.basename(image_path)}: Mostly monochrome (B:{black_ratio:.2f}, W:{white_ratio:.2f}, NW:{near_white_ratio:.2f})")
                return False
            
            # 5. Simple edge detection using PIL filters (alternative to scipy)
            # Convert to grayscale for edge detection
            gray_img = img.convert('L')
            gray_array = np.array(gray_img)
            
            # Simple edge detection using gradient calculation
            # Calculate horizontal and vertical gradients
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            
            # Pad to match original dimensions
            grad_x_padded = np.pad(grad_x, ((0, 0), (0, 1)), mode='constant')
            grad_y_padded = np.pad(grad_y, ((0, 1), (0, 0)), mode='constant')
            
            # Combine gradients
            edges = grad_x_padded + grad_y_padded
            
            # Calculate edge density
            edge_threshold = np.mean(edges) + np.std(edges)
            edge_pixels = np.sum(edges > edge_threshold)
            edge_ratio = edge_pixels / total_pixels
            
            if edge_ratio > max_text_ratio and unique_colors < 30:  # Only reject high-edge images if they also have few colors
                print(f"❌ Rejected {os.path.basename(image_path)}: Likely text/simple graphics (edge ratio: {edge_ratio:.2f}, colors: {unique_colors})")
                return False
            
            # 6. Content complexity check
            # Calculate variance in pixel values as a measure of image complexity
            variance = np.var(img_array)
            if variance < 100:  # Very low variance suggests simple/uniform content
                print(f"❌ Rejected {os.path.basename(image_path)}: Low content complexity (variance: {variance:.2f})")
                return False
            
            # 7. Check for dominant single colors (emoji-like images)
            # Count how many pixels are the most common color
            from collections import Counter
            # Convert pixels to tuples for proper counting
            pixel_tuples = [tuple(pixel) for pixel in pixels]
            color_counts = Counter(pixel_tuples)
            most_common_color_count = color_counts.most_common(1)[0][1]
            dominant_color_ratio = most_common_color_count / total_pixels
            
            # More lenient for research figures - they often have white backgrounds
            # Only reject if it's a simple image (few colors) AND highly dominant color
            if dominant_color_ratio > 0.95 and unique_colors < 15:
                print(f"❌ Rejected {os.path.basename(image_path)}: Dominant single color ({dominant_color_ratio:.2f}) with few colors ({unique_colors})")
                return False
            
            print(f"✅ Accepted {os.path.basename(image_path)}: Quality image ({width}x{height}, {unique_colors} colors, edge:{edge_ratio:.2f}, var:{variance:.0f})")
            return True
            
    except Exception as e:
        print(f"❌ Error analyzing {image_path}: {e}")
        return False

def extract_images_and_create_gif(pdf_path, image_dir, entry_id, gif_path, duration=3000, size=(512, 512), transition_frames=8):
    """
    Extracts high-quality images from a PDF and creates a Twitter-compatible GIF file.
    Now includes intelligent filtering to remove low-quality images like emojis, symbols, or text.
    
    Twitter GIF requirements:
    - Max 15MB file size
    - Max 512x512 resolution (or 1280x1080 for landscape)
    - Infinite loop
    - Optimized for web
    - Smooth, appealing transitions
    """
    all_images = []
    quality_images = []
    
    # First pass: Extract all images
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
                all_images.append(image_path)
                pix = None  # Clean up Pixmap object
                print(f"📷 Extracted: {os.path.basename(image_path)}")
                
                if len(all_images) >= 12:  # Extract more initially to have options after filtering
                    break
            if len(all_images) >= 12:
                break
    
    print(f"\n🔍 Analyzing {len(all_images)} extracted images for quality...")
    
    # Second pass: Filter for quality images
    for image_path in all_images:
        if is_quality_image(image_path):
            quality_images.append(image_path)
        else:
            # Remove low-quality images to save space
            try:
                os.remove(image_path)
            except:
                pass
    
    print(f"\n✨ Selected {len(quality_images)} quality images out of {len(all_images)} extracted")
    
    # Limit to best 6 images for optimal viewing
    if len(quality_images) > 6:
        quality_images = quality_images[:6]
        print(f"📊 Using top {len(quality_images)} images for GIF")

    if quality_images:
        # Create a Twitter-compatible GIF from the filtered images
        gif_images = []
        
        for i in range(len(quality_images)):
            img = Image.open(quality_images[i])
            img = ImageOps.pad(img, size, color="white")  # Use white padding for better visibility
            
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Hold each image for longer (multiple frames)
            hold_frames = 8  # Show each image for 8 frames before transitioning
            for _ in range(hold_frames):
                gif_images.append(img.copy())
            
            # Add smooth transition to next image (if not the last image)
            if i < len(quality_images) - 1:
                next_img = Image.open(quality_images[i + 1])
                next_img = ImageOps.pad(next_img, size, color="white")
                if next_img.mode != 'RGB':
                    next_img = next_img.convert('RGB')
                
                # Create smooth transition frames
                for j in range(1, transition_frames + 1):
                    alpha = j / (transition_frames + 1)
                    # Use a smoother easing function for more natural transitions
                    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # Smoothstep function
                    blend = Image.blend(img, next_img, smooth_alpha)
                    gif_images.append(blend)
        
        # Hold the last image for a bit longer before looping
        final_hold_frames = 12
        for _ in range(final_hold_frames):
            gif_images.append(gif_images[-1].copy())

        # Calculate frame duration for smooth, appealing playback
        total_frames = len(gif_images)
        frame_duration = max(120, duration // total_frames)  # Min 120ms per frame for smoother playback
        
        print(f"📹 Creating GIF with {total_frames} frames, {frame_duration}ms per frame")
        
        # Save GIF with Twitter-optimized settings
        gif_images[0].save(
            gif_path, 
            save_all=True, 
            append_images=gif_images[1:], 
            duration=frame_duration,
            loop=0,  # Infinite loop
            optimize=True,  # Optimize for file size
            disposal=2  # Clear frame before next one
        )
        
        # Check file size and reduce quality if too large
        file_size = os.path.getsize(gif_path)
        if file_size > 10 * 1024 * 1024:  # If larger than 10MB
            print(f"⚠️  GIF too large ({file_size/1024/1024:.1f}MB), creating optimized version...")
            
            # Create a smaller, more optimized version
            optimized_images = []
            step = max(1, len(gif_images) // 20)  # Reduce to max 20 frames
            for i in range(0, len(gif_images), step):
                img = gif_images[i].resize((400, 400), Image.Resampling.LANCZOS)
                optimized_images.append(img)
            
            optimized_images[0].save(
                gif_path,
                save_all=True,
                append_images=optimized_images[1:],
                duration=max(200, duration // len(optimized_images)),
                loop=0,
                optimize=True,
                disposal=2
            )
        
        final_size = os.path.getsize(gif_path)
        print(f"🎬 GIF created and saved to: {gif_path} ({final_size/1024/1024:.1f}MB)")
        return gif_path
    else:
        print("❌ No quality images found in PDF after filtering.")
        return None

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

def smart_paper_search(days=1, max_fetch=100, target_papers=10):
    """
    Intelligent paper search that fetches more papers, scores them for engagement,
    and selects the best diverse set for non-technical audiences.
    """
    # Define different research topics/areas to search
    research_topics = [
        "cs.AI OR cs.LG OR cs.CV OR cs.CL",  # AI/ML/Computer Vision/NLP
        "cs.RO OR cs.HC OR cs.CY",          # Robotics/Human-Computer Interaction/Cybersecurity  
        "q-bio OR physics.bio-ph",          # Biology/Biophysics
        "econ OR q-fin",                    # Economics/Finance
        "stat.ML OR math.OC",               # Statistics/Optimization
        "cs.DC OR cs.NI OR cs.CR",           # Distributed Computing/Networking/Cryptography
        "astro-ph OR gr-qc OR hep-th",      # Astronomy/General Relativity/High Energy Physics
        "cond-mat OR quant-ph",             # Condensed Matter/Quantum Physics
        "math.PR OR math.ST",                # Probability/Statistics
        "cs.SI OR cs.SE OR cs.CE"           # Social and Information Networks/Software Engineering/Computational Engineering
    ]
    
    # Start with the specified time range
    search_attempts = 0
    all_papers = []
    
    for current_topic in research_topics:  # Ensure we have enough papers to choose from
        
        start_date = (datetime.today() - timedelta(days=days)).strftime('%Y%m%d')
        end_date = datetime.today().strftime('%Y%m%d')
                
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
        
        print(f"Found {len(results)} papers ({len(new_papers)} new) in {current_topic} (total: {len(all_papers)})")
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

def cleanup_paper_files(pdf_path, image_paths=None, keep_pdfs=False):
    """
    Clean up PDF and image files after processing.
    
    Args:
        pdf_path (str): Path to the PDF file
        image_paths (list): List of image file paths to remove
        keep_pdfs (bool): If True, keep PDF files (only remove images)
    """
    files_removed = []
    
    try:
        # Remove PDF file unless keep_pdfs is True
        if not keep_pdfs and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            files_removed.append(pdf_path)
            print(f"🗑️  Removed PDF: {os.path.basename(pdf_path)}")
        
        # Remove image files
        if image_paths:
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                    files_removed.append(image_path)
                    print(f"🗑️  Removed image: {os.path.basename(image_path)}")
        
        # Also look for and remove any generated GIF
        if pdf_path:
            gif_path = pdf_path.replace('.pdf', '.gif').replace('arxiv_papers', 'arxiv_images')
            if os.path.exists(gif_path):
                os.remove(gif_path)
                files_removed.append(gif_path)
                print(f"🗑️  Removed GIF: {os.path.basename(gif_path)}")
                
    except Exception as e:
        print(f"⚠️  Error cleaning up files: {e}")
    
    return files_removed

def get_generated_image_paths(pdf_path, image_dir):
    """
    Get paths of all images that might have been generated from a PDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        image_dir (str): Directory where images are stored
        
    Returns:
        list: List of image file paths
    """
    entry_id = os.path.basename(pdf_path).replace('.pdf', '')
    image_pattern = os.path.join(image_dir, f"{entry_id}_image_*")
    
    # Find all matching image files
    import glob
    image_paths = glob.glob(image_pattern)
    
    # Also check for the GIF file
    gif_path = os.path.join(image_dir, f"{entry_id}.gif")
    if os.path.exists(gif_path):
        image_paths.append(gif_path)
    
    return image_paths

def cleanup_old_files(pdf_dir="arxiv_papers", image_dir="arxiv_images", days_old=7):
    """
    Clean up files older than specified days.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        image_dir (str): Directory containing image files  
        days_old (int): Remove files older than this many days
    """
    import glob
    import time
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    removed_count = 0
    
    # Clean old PDFs
    if os.path.exists(pdf_dir):
        for pdf_file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
            if os.path.getmtime(pdf_file) < cutoff_time:
                os.remove(pdf_file)
                removed_count += 1
                print(f"🗑️  Removed old PDF: {os.path.basename(pdf_file)}")
    
    # Clean old images and GIFs
    if os.path.exists(image_dir):
        for pattern in ["*.png", "*.jpg", "*.gif"]:
            for image_file in glob.glob(os.path.join(image_dir, pattern)):
                if os.path.getmtime(image_file) < cutoff_time:
                    os.remove(image_file)
                    removed_count += 1
                    print(f"🗑️  Removed old image: {os.path.basename(image_file)}")
    
    print(f"✅ Cleanup complete! Removed {removed_count} old files.")
    return removed_count

def upload_media_to_twitter(api, media_path):
    """
    Upload media to Twitter with proper handling for different file types.
    
    Args:
        api: Twitter API instance
        media_path (str): Path to the media file
        
    Returns:
        str: Media ID if successful, None if failed
    """
    try:
        if not os.path.exists(media_path):
            print(f"⚠️  Media file not found: {media_path}")
            return None
        
        file_size = os.path.getsize(media_path)
        print(f"📁 Uploading {os.path.basename(media_path)} ({file_size / 1024 / 1024:.1f}MB)")
        
        # Check file type and size
        if media_path.lower().endswith('.gif'):
            # For GIF files, use specific Twitter requirements
            if file_size > 15 * 1024 * 1024:  # 15MB limit for GIFs
                print(f"⚠️  GIF file too large ({file_size / 1024 / 1024:.1f}MB), max 15MB")
                return None
            
            # Use chunked upload for GIFs
            media = api.chunked_upload(
                filename=media_path,
                media_category='tweet_gif',
                additional_owners=None
            )
        else:
            # For images (PNG, JPG)
            if file_size > 5 * 1024 * 1024:  # 5MB limit for images
                print(f"⚠️  Image file too large ({file_size / 1024 / 1024:.1f}MB), max 5MB")
                return None
            
            media = api.media_upload(media_path)
        
        print(f"✅ Media uploaded successfully: {os.path.basename(media_path)}")
        return media.media_id
        
    except Exception as e:
        print(f"⚠️  Failed to upload media {media_path}: {e}")
        print(f"🔄 Trying alternative upload method...")
        
        # Fallback: try regular upload even for GIFs
        try:
            media = api.media_upload(media_path)
            print(f"✅ Media uploaded with fallback method: {os.path.basename(media_path)}")
            return media.media_id
        except Exception as e2:
            print(f"❌ All upload methods failed: {e2}")
            return None

def tweet_arxiv_papers(debug=False, days=1, max_results=6, enable_author_tagging=True, use_smart_selection=True, cleanup_files=True, keep_pdfs=False):
    """
    Main function to fetch, process, and tweet arXiv papers.
    
    Args:
        debug (bool): If True, doesn't post to Twitter, just prints output
        days (int): How many days back to search for papers
        max_results (int): Target number of papers to process
        enable_author_tagging (bool): If True, attempts to tag paper authors
        use_smart_selection (bool): If True, uses AI-powered paper selection
        cleanup_files (bool): If True, removes files after processing
        keep_pdfs (bool): If True and cleanup_files=True, keeps PDFs but removes images
    """
    # Authenticate to Twitter using selected method
    if not debug:
        if TWITTER_AUTH_METHOD == "twikit":
            print(f"🔄 Using twikit authentication method (better rate limits)")
            # Twikit requires async, we'll handle this in the tweet sending function
        else:
            print(f"🔄 Using tweepy authentication method (traditional API)")
            success = setup_tweepy_client()
            if not success:
                print("❌ Authentication failed, exiting...")
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
        # Post the main tweet to start the thread (with unique variation to avoid duplicates)
        thread_opener = TweetVariations.get_unique_opener()
        main_tweet_id = send_tweet_sync(thread_opener)
        if not main_tweet_id:
            print("❌ Failed to send main tweet, exiting...")
            return

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
            # Clean up the corrupted PDF file
            if cleanup_files and os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"🗑️  Removed corrupted PDF: {os.path.basename(pdf_path)}")
            continue


        # Extract the first image from the PDF
        #image_path = extract_relevant_image_from_pdf(pdf_path, image_dir, result.entry_id.split('/')[-1])
        image_path = extract_images_and_create_gif(pdf_path, image_dir, result.entry_id.split('/')[-1], f"{image_dir}/{result.entry_id.split('/')[-1]}.gif")
        
        # Generate tweet text using ChatGPT
        prompt = PromptTemplates.get_summary_prompt(text, style="viral")
        explanation = get_text(prompt)
        
        # Format tweet with author tags
        base_tweet = f"{explanation} Source: {result.entry_id}"
        print(f"🔍 DEBUG - Base tweet: {base_tweet}")
        print(f"🔍 DEBUG - About to call format_tweet_with_author_tags with:")
        print(f"  - Authors: {[a.name for a in result.authors]}")
        print(f"  - Handles: {author_handles}")
        
        final_tweet = format_tweet_with_author_tags(base_tweet, result.authors, author_handles)
        
        print(f"🔍 DEBUG - Final tweet result: {final_tweet}")
        print(f"🔍 DEBUG - Final tweet length: {len(final_tweet)}")
        print(f"{final_tweet}\n")

        # Post the explanation with an image if available
        if not debug:
            media_path_list = [image_path] if image_path and os.path.exists(image_path) else None
            tweet_id = send_tweet_sync(final_tweet, media_path_list, reply_to_id=main_tweet_id)
            if not tweet_id:
                print("❌ Failed to send reply tweet")

        # Clean up files after processing (if enabled)
        if cleanup_files:
            # Get all generated image paths for this paper
            generated_images = get_generated_image_paths(pdf_path, image_dir)
            cleanup_paper_files(pdf_path, generated_images, keep_pdfs=keep_pdfs)

    if not debug:
        # Final tweet to close the thread
        final_tweet_id = send_tweet_sync(
            TweetVariations.get_unique_closer(),
            reply_to_id=main_tweet_id
        )
        if not final_tweet_id:
            print("❌ Failed to send closing tweet")

# Run the function
if __name__ == "__main__":
    tweet_arxiv_papers(
        debug=False, 
        enable_author_tagging=True, 
        days=5, 
        max_results=10,
        use_smart_selection=True,
        cleanup_files=True,  # Clean up files after processing
        keep_pdfs=False      # Remove both PDFs and images
    )
