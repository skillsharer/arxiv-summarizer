import arxiv
import fitz  # PyMuPDF
import os
import requests
import tweepy
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
    except Exception as e:
        response = ""
        print(f"An error occurred: {e}")

    return response

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

def tweet_arxiv_papers(debug=False, days=1, max_results=10):
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

    # Set the search period
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y%m%d')
    end_date = datetime.today().strftime('%Y%m%d')

    # Define the search query
    search_query = arxiv.Search(
        query=f"submittedDate:[{start_date}0000 TO {end_date}2359]",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # Create a client instance and folders for PDFs and images
    arxiv_client = arxiv.Client()
    pdf_dir = "arxiv_papers"
    image_dir = "arxiv_images"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Retrieve and process the results
    results = list(arxiv_client.results(search_query))
    print(f"Number of results: {len(results)}")

    if not results:
        print("No results found for the specified date range. Please check your query and try again.")
        return

    if not debug:
        # Post the main tweet to start the thread
        main_tweet = client.create_tweet(
            text="🌐 What's trending in research? Check out today’s top arXiv discoveries, hand-picked by AI! Ready to expand your horizon? 📈 Learn more in the thread:👇🏻 #AI #MachineLearning #Innovation #Research #Science #Physics #Chemistry #Biology"
        )
        main_tweet_id = main_tweet.data['id']  # Store the main tweet ID

    for result in results:
        print(f"Processing: {result.title}")

        # Download the PDF
        pdf_path = os.path.join(pdf_dir, f"{result.entry_id.split('/')[-1]}.pdf")
        result.download_pdf(filename=pdf_path)

        # Extract text from the PDF
        text = ""
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()

        # Extract the first image from the PDF
        image_path = extract_first_image_from_pdf(pdf_path, image_dir, result.entry_id.split('/')[-1])

        # Generate tweet text using ChatGPT
        prompt = (
            "Craft a tweet-sized summary of this article, packed with 5+ relevant hashtags to maximize reach. Make it accessible and engaging for readers unfamiliar with the topic. Simplify the research findings, highlighting how the results could directly impact our daily lives and future. " + text
        )
        explanation = get_text(prompt)
        print(f"{explanation} Source: {result.entry_id}\n")

        # Post the explanation with an image if available
        if not debug:
            media_ids = []
            if image_path:
                media = api.media_upload(image_path)
                media_ids.append(media.media_id)

            client.create_tweet(
                text=f"{explanation} Source: {result.entry_id}",
                in_reply_to_tweet_id=main_tweet_id,
                media_ids=media_ids if media_ids else None
            )

    if not debug:
        # Final tweet to close the thread
        client.create_tweet(
            text="🚀 That’s a wrap on today’s highlights! Make sure to follow for daily updates on the latest discoveries in science, tech, and beyond. Join our community of curious minds! 🌐✨ #AI #ResearchBuzz #StayUpdated #InnovationDaily",
            in_reply_to_tweet_id=main_tweet_id
        )

# Run the function
tweet_arxiv_papers(debug=False)
