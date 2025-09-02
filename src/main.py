"""
ArXiv Summarizer - Main Application
Fetches, processes, and tweets arXiv papers with AI-generated summaries
"""

import os
import re

import fitz
from pymupdf import FileDataError

from author_lookup import get_twitter_handles_enhanced

# Import modularized components
from config import TWITTER_AUTH_METHOD
from openai_client import get_text
from paper_search import smart_paper_search, traditional_paper_search
from pdf_processor import extract_images_and_create_gif
from prompts import MessageTemplates, PromptTemplates, TweetVariations
from twitter_auth import setup_tweepy_client
from twitter_client import format_tweet_with_author_tags, send_tweet_sync
from utils import cleanup_paper_files, get_generated_image_paths


def tweet_arxiv_papers(
    debug=False,
    days=1,
    max_results=6,
    enable_author_tagging=True,
    use_smart_selection=True,
    cleanup_files=True,
    keep_pdfs=False,
):
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
        results = traditional_paper_search(days=days, max_results=max_results)

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
        print(
            MessageTemplates.format_message(MessageTemplates.PROCESSING_PAPER, title=result.title)
        )
        print(
            MessageTemplates.format_message(
                MessageTemplates.EXTRACTING_AUTHORS,
                authors=[author.name for author in result.authors],
            )
        )

        # Get Twitter handles for authors using enhanced method
        author_handles = {}
        if enable_author_tagging:
            # Extract arXiv ID from the entry_id (format: http://arxiv.org/abs/2024.12345v1)
            arxiv_id = result.entry_id.split("/")[-1]
            # Remove version number (v1, v2, etc.) to get clean arXiv ID
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            print(
                MessageTemplates.format_message(
                    MessageTemplates.ARXIV_ID_EXTRACTED, arxiv_id=arxiv_id
                )
            )
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

        # Extract images and create GIF
        image_path = extract_images_and_create_gif(
            pdf_path,
            image_dir,
            result.entry_id.split("/")[-1],
            f"{image_dir}/{result.entry_id.split('/')[-1]}.gif",
        )

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
            TweetVariations.get_unique_closer(), reply_to_id=main_tweet_id
        )
        if not final_tweet_id:
            print("❌ Failed to send closing tweet")


# Run the function
if __name__ == "__main__":
    tweet_arxiv_papers(
        debug=True,
        enable_author_tagging=True,
        days=5,
        max_results=10,
        use_smart_selection=True,
        cleanup_files=False,  # Clean up files after processing
        keep_pdfs=True,  # Remove both PDFs and images
    )
