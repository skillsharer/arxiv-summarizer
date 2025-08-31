"""
Utility functions module
Handles file management, cleanup, and media upload functionality
"""

import glob
import os
import time

import tweepy


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
            gif_path = pdf_path.replace(".pdf", ".gif").replace("arxiv_papers", "arxiv_images")
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
    entry_id = os.path.basename(pdf_path).replace(".pdf", "")
    image_pattern = os.path.join(image_dir, f"{entry_id}_image_*")

    # Find all matching image files
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
        if media_path.lower().endswith(".gif"):
            # For GIF files, use specific Twitter requirements
            if file_size > 15 * 1024 * 1024:  # 15MB limit for GIFs
                print(f"⚠️  GIF file too large ({file_size / 1024 / 1024:.1f}MB), max 15MB")
                return None

            # Use chunked upload for GIFs
            media = api.chunked_upload(
                filename=media_path, media_category="tweet_gif", additional_owners=None
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
