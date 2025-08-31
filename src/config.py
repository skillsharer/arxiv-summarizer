"""
Configuration module for ArXiv Summarizer
Handles environment variables and global settings
"""

import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_BEARER_TOKEN = os.getenv("OPENAI_BEARER_TOKEN")

# Twitter API Configuration
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

# Global variables for Twitter clients - initialized as None
twitter_client = None
twitter_api = None
twikit_client = None
