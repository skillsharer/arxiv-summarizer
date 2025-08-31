"""
Twitter client module
Handles tweet sending functionality for both twikit and tweepy methods
"""

import asyncio
import concurrent.futures
import os

from config import TWITTER_AUTH_METHOD, twikit_client, twitter_api, twitter_client
from twitter_auth import handle_twitter_auth_failure, setup_tweepy_client, setup_twikit_client


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
                    media_id = await twikit_client.upload_media(media_path)
                    media_ids.append(media_id)

        # Send tweet with reply if specified
        if reply_to_id:
            if media_ids:
                tweet = await twikit_client.create_tweet(
                    text=text, media_ids=media_ids, reply_to=reply_to_id
                )
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
        if (
            "401" in error_msg or "Could not authenticate" in error_msg or 'code":32' in error_msg
        ) and retry_auth:
            print(f"🔄 Authentication error detected, attempting fresh login...")
            try:
                # Force re-authentication
                success = await setup_twikit_client(force_reauth=True)
                if success:
                    return await send_tweet_twikit(text, media_paths, reply_to_id, retry_auth=False)
                else:
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
                tweet = twitter_client.create_tweet(
                    text=text, media_ids=media_ids, in_reply_to_tweet_id=reply_to_id
                )
            else:
                tweet = twitter_client.create_tweet(text=text, in_reply_to_tweet_id=reply_to_id)
        else:
            if media_ids:
                tweet = twitter_client.create_tweet(text=text, media_ids=media_ids)
            else:
                tweet = twitter_client.create_tweet(text=text)

        print(f"📤 Tweet sent successfully via tweepy (ID: {tweet.data['id']})")
        return tweet.data["id"]

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
