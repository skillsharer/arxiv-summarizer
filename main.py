import arxiv
import fitz  # PyMuPDF
import os
import requests
import tweepy
from dotenv import load_dotenv
from datetime import datetime, timedelta
from datetime import datetime, timedelta
load_dotenv()

debug = False  # Set to True to run in debug mode, which doesn't post to Twitter

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
        print("Successfully authenticated to Twitter.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Set the search period to the past 1 day(s)
start_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
end_date = datetime.today().strftime('%Y%m%d')

# Define the search query for all arXiv categories with today's date
search_query = arxiv.Search(
    query=f"submittedDate:[{start_date}0000 TO {end_date}2359]",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

# Create a client instance
arxiv_client = arxiv.Client()

# Directory to save PDFs
pdf_dir = "arxiv_papers"
os.makedirs(pdf_dir, exist_ok=True)

# Retrieve and process the results
results = list(arxiv_client.results(search_query))
print(f"Number of results: {len(results)}")

if not results:
    print("No results found for today's date. Please check your query and try again.")
else:
    if not debug:
        # Post the main tweet to start the thread
        main_tweet = client.create_tweet(
            text="🚀 Ready to level up your knowledge? Dive into today’s top research findings from arXiv, hand-picked by AI!💡 #NewResearch #ArxivNews #ScienceDaily #AIInsights #MachineLearning #Breakthroughs #DataScienceDaily #FutureReady #TechResearch"
        )
        main_tweet_id = main_tweet.data['id']  # Store the main tweet ID

    for result in results:
        print(f"Processing: {result.title}")

        # Download the PDF
        pdf_path = os.path.join(pdf_dir, f"{result.entry_id.split('/')[-1]}.pdf")
        result.download_pdf(filename=pdf_path)

        # Extract text from the PDF
        with fitz.open(pdf_path) as pdf_document:
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()

        # Save the extracted text to a file
        text_path = pdf_path.replace(".pdf", ".txt")
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        print(f"Text extracted and saved to: {text_path}\n")

        # Explain article with ChatGPT API
        prompt = (
            "Summarize this article in a single tweet (under 280 characters) and include at least 5 relevant hashtags "
            "to boost reach. Make it easy to understand and engaging for readers who are not experts in the topic. "
            "Explain the research as simple as you can and elaborate on how the research results could impact our lives. " + text
        )

        # Call the ChatGPT API to generate an explanation
        explanation = get_text(prompt)
        print(f"{explanation} Source: {result.entry_id}\n")

        if not debug:
            # Tweet the explanation as a reply in the thread
            client.create_tweet(text=f"{explanation} Source: {result.entry_id}", in_reply_to_tweet_id=main_tweet_id)

    if not debug:
        # Final tweet to close the thread
        client.create_tweet(
            text="🚀 That’s a wrap on today’s highlights! Make sure to follow for daily updates on the latest discoveries in science, tech, and beyond. Join our community of curious minds! 🌐✨ #AI #ResearchBuzz #StayUpdated #InnovationDaily",
            in_reply_to_tweet_id=main_tweet_id
        )
