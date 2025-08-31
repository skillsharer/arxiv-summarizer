"""
PDF processing module
Handles PDF text extraction, image extraction, and GIF creation
"""

import os
from collections import Counter

import fitz
import numpy as np
from PIL import Image, ImageOps
from pymupdf import FileDataError


def extract_relevant_image_from_pdf(pdf_path, image_dir, entry_id):
    """Extracts a relevant image from a PDF by analyzing surrounding text, checking for keywords in captions,
    and filtering based on color and size criteria."""

    # Keywords indicating potentially relevant images
    keywords = ["figure", "graph", "diagram", "illustration", "results", "method", "conclusion"]
    image_path = None

    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text(
                "text"
            ).lower()  # Get text content in lowercase for keyword matching

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

                # 1. Apply size threshold to skip small or irrelevant images before expensive analysis
                if img_data.width < 200 or img_data.height < 200:
                    pix = None
                    continue

                # 2. Filter out black/blank images
                img_array = np.array(img_data)
                black_pixels_ratio = np.mean(np.all(img_array == [0, 0, 0], axis=2))

                if black_pixels_ratio > 0.8:  # Skip black/blank images
                    pix = None
                    continue

                # Resize the image while maintaining aspect ratio
                max_size = (800, 800)  # Define the maximum size for the image
                img_data.thumbnail(max_size, Image.Resampling.NEAREST)

                # 3. Check for relevant keywords around image in the page text
                # Find the first occurrence of any keyword in the page text
                first_pos = -1
                for keyword in keywords:
                    pos = page_text.find(keyword)
                    if pos != -1 and (first_pos == -1 or pos < first_pos):
                        first_pos = pos
                if first_pos == -1:
                    pix = None
                    continue
                surrounding_text = page_text[max(0, first_pos - 100) : first_pos + 100]
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
            if img.mode != "RGB":
                img = img.convert("RGB")

            width, height = img.size

            # 1. Size filter - reject tiny images (likely icons, emojis, or symbols)
            if width < min_size[0] or height < min_size[1]:
                print(f"❌ Rejected {os.path.basename(image_path)}: Too small ({width}x{height})")
                return False

            # 2. Aspect ratio filter - reject extremely narrow or wide images
            aspect_ratio = width / height
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Bad aspect ratio ({aspect_ratio:.2f})"
                )
                return False

            # Convert to numpy array for analysis
            img_array = np.array(img)

            # 3. Color diversity filter - reject images with too few colors (likely simple graphics/text)
            # Reshape to 2D array of RGB values and count unique colors more efficiently
            pixels = img_array.reshape(-1, 3)
            # Convert to tuples for proper hashing
            unique_colors = len(set(map(tuple, pixels)))

            if unique_colors < min_colors:
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Too few colors ({unique_colors})"
                )
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
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Mostly monochrome (B:{black_ratio:.2f}, W:{white_ratio:.2f}, NW:{near_white_ratio:.2f})"
                )
                return False

            # 5. Simple edge detection using PIL filters (alternative to scipy)
            # Convert to grayscale for edge detection
            gray_img = img.convert("L")
            gray_array = np.array(gray_img)

            # Simple edge detection using gradient calculation
            # Calculate horizontal and vertical gradients
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))

            # Pad to match original dimensions
            grad_x_padded = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant")
            grad_y_padded = np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")

            # Combine gradients
            edges = grad_x_padded + grad_y_padded

            # Calculate edge density
            edge_threshold = np.mean(edges) + np.std(edges)
            edge_pixels = np.sum(edges > edge_threshold)
            edge_ratio = edge_pixels / total_pixels

            if (
                edge_ratio > max_text_ratio and unique_colors < 30
            ):  # Only reject high-edge images if they also have few colors
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Likely text/simple graphics (edge ratio: {edge_ratio:.2f}, colors: {unique_colors})"
                )
                return False

            # 6. Content complexity check
            # Calculate variance in pixel values as a measure of image complexity
            variance = np.var(img_array)
            if variance < 100:  # Very low variance suggests simple/uniform content
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Low content complexity (variance: {variance:.2f})"
                )
                return False

            # 7. Check for dominant single colors (emoji-like images)
            # Count how many pixels are the most common color
            # Convert pixels to tuples for proper counting
            pixel_tuples = [tuple(pixel) for pixel in pixels]
            color_counts = Counter(pixel_tuples)
            most_common_color_count = color_counts.most_common(1)[0][1]
            dominant_color_ratio = most_common_color_count / total_pixels

            # More lenient for research figures - they often have white backgrounds
            # Only reject if it's a simple image (few colors) AND highly dominant color
            if dominant_color_ratio > 0.95 and unique_colors < 15:
                print(
                    f"❌ Rejected {os.path.basename(image_path)}: Dominant single color ({dominant_color_ratio:.2f}) with few colors ({unique_colors})"
                )
                return False

            print(
                f"✅ Accepted {os.path.basename(image_path)}: Quality image ({width}x{height}, {unique_colors} colors, edge:{edge_ratio:.2f}, var:{variance:.0f})"
            )
            return True

    except Exception as e:
        print(f"❌ Error analyzing {image_path}: {e}")
        return False


def extract_images_and_create_gif(
    pdf_path, image_dir, entry_id, gif_path, duration=3000, size=(512, 512), transition_frames=8
):
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
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Hold each image for longer (multiple frames)
            hold_frames = 8  # Show each image for 8 frames before transitioning
            for _ in range(hold_frames):
                gif_images.append(img.copy())

            # Add smooth transition to next image (if not the last image)
            if i < len(quality_images) - 1:
                next_img = Image.open(quality_images[i + 1])
                next_img = ImageOps.pad(next_img, size, color="white")
                if next_img.mode != "RGB":
                    next_img = next_img.convert("RGB")

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
        frame_duration = max(
            120, duration // total_frames
        )  # Min 120ms per frame for smoother playback

        print(f"📹 Creating GIF with {total_frames} frames, {frame_duration}ms per frame")

        # Save GIF with Twitter-optimized settings
        gif_images[0].save(
            gif_path,
            save_all=True,
            append_images=gif_images[1:],
            duration=frame_duration,
            loop=0,  # Infinite loop
            optimize=True,  # Optimize for file size
            disposal=2,  # Clear frame before next one
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
                disposal=2,
            )

        final_size = os.path.getsize(gif_path)
        print(f"🎬 GIF created and saved to: {gif_path} ({final_size/1024/1024:.1f}MB)")
        return gif_path
    else:
        print("❌ No quality images found in PDF after filtering.")
        return None
