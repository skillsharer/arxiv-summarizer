"""
Twitter authentication module
Handles both twikit and tweepy authentication methods
"""

import asyncio
import os

import tweepy

from config import (
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET,
    BEARER_TOKEN,
    CONSUMER_KEY,
    CONSUMER_SECRET,
    TWITTER_AUTH_METHOD,
    TWITTER_EMAIL,
    TWITTER_PASSWORD,
    TWITTER_USERNAME,
    twikit_client,
    twitter_api,
    twitter_client,
)


async def setup_twikit_client(force_reauth=False):
    """Setup and authenticate twikit client with session persistence and validation"""
    global twikit_client

    try:
        from twikit import Client

        twikit_client = Client("en-US")
        cookies_file = "twitter_cookies.json"

        # Try to load existing cookies unless forced to re-authenticate
        if not force_reauth and os.path.exists(cookies_file):
            try:
                twikit_client.load_cookies(cookies_file)
                print("🍪 Loaded existing Twitter session cookies")

                # Validate the session by trying a simple operation
                print("🔍 Validating session...")
                try:
                    # Try to get user info to validate session
                    user = await twikit_client.get_user_by_screen_name(
                        TWITTER_USERNAME or "twitter"
                    )
                    if user:
                        print("✅ Session validation successful")
                        return True
                    else:
                        print("⚠️  Session validation failed, fresh login required")
                except Exception as e:
                    print(f"⚠️  Session validation failed: {e}")

            except Exception as e:
                print(f"⚠️  Failed to load cookies: {e}")

        # Fresh login required
        if TWITTER_USERNAME and TWITTER_PASSWORD:
            print("🔐 Performing fresh login to Twitter with twikit...")
            try:
                await twikit_client.login(
                    auth_info_1=TWITTER_USERNAME,
                    auth_info_2=TWITTER_EMAIL or TWITTER_USERNAME,
                    password=TWITTER_PASSWORD,
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
    cookies_file = "twitter_cookies.json"
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
    print("\n" + "=" * 60)
    print("🚨 TWITTER AUTHENTICATION ISSUE DETECTED")
    print("=" * 60)
    print("Twitter has temporarily restricted login due to 'unusual activity'.")
    print("This is common when using automation tools.")
    print("\n📋 SOLUTIONS (try in order):")
    print("1. ⏰ WAIT 15-30 minutes and try again")
    print("2. 🌐 Log into Twitter manually in browser first")
    print("3. 🔄 Switch to tweepy method temporarily:")
    print("   Add to .env: TWITTER_AUTH_METHOD=tweepy")
    print("4. 🧹 Clear session and retry:")
    print('   python -c "import twitter_auth; twitter_auth.clear_twitter_session()"')
    print("5. 🐛 Test with debug mode:")
    print('   python -c "import main; main.tweet_arxiv_papers(debug=True)"')
    print("\n💡 TIP: Tweepy method is more stable for production use!")
    print("=" * 60 + "\n")


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
            wait_on_rate_limit=True,
        )
        auth = tweepy.OAuth1UserHandler(
            CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
        )
        twitter_api = tweepy.API(auth)
        print("✅ Successfully authenticated with tweepy")
        return True
    except Exception as e:
        print(f"❌ Tweepy authentication failed: {e}")
        return False
