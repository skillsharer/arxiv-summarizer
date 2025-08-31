# arXiv Paper Summarizer and Twitter Bot

This script fetches the latest research papers from arXiv, summarizes them using OpenAI's GPT model, and posts the summaries to Twitter as a thread. The script is designed to run daily and provide a concise overview of the latest research across various fields.

## Features

- Fetches the latest research papers from arXiv.
- Downloads and extracts text from the PDFs.
- Summarizes the papers using OpenAI's GPT model.
- Posts the summaries to Twitter as a thread.
- **Author Tagging**: Automatically tags paper authors on Twitter using arXiv → ORCID → Twitter pipeline.
- **Modular Prompts**: Separated prompts and templates for easy customization.

## Requirements

- Python 3.8+
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Setup

1. **Clone the repository:**

    ```sh
    git clone git@github.com:skillsharer/arxiv-summarizer.git
    cd arxiv-twitter-bot
    ```

2. **Create and activate a conda environment:**

    ```sh
    conda create --name arxiv-twitter-bot python=3.12
    conda activate arxiv-twitter-bot
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Create a .env file in the root directory and add your API keys:**

    **Option 1: Traditional Twitter API (Tweepy)**
    ```env
    OPENAI_BEARER_TOKEN=your_openai_bearer_token
    
    # Twitter API Configuration
    TWITTER_AUTH_METHOD=tweepy
    CONSUMER_KEY=your_twitter_consumer_key
    CONSUMER_SECRET=your_twitter_consumer_secret
    ACCESS_TOKEN=your_twitter_access_token
    ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
    BEARER_TOKEN=your_twitter_bearer_token
    ```

    **Option 2: High-Rate-Limit Method (Twikit) - Recommended**
    ```env
    OPENAI_BEARER_TOKEN=your_openai_bearer_token
    
    # Twitter Configuration (Better Rate Limits)
    TWITTER_AUTH_METHOD=twikit
    TWITTER_USERNAME=your_twitter_username
    TWITTER_PASSWORD=your_twitter_password
    TWITTER_EMAIL=your_twitter_email  # optional
    ```

## Twitter Authentication Methods

This bot supports two Twitter authentication methods:

### Method 1: Traditional API (Tweepy)
- **Pros**: Stable, officially supported
- **Cons**: Lower rate limits (300 tweets/15min, 50 tweets/hour with media)
- **Setup**: Requires Twitter API keys
- **Use Case**: Production applications requiring stability

### Method 2: High-Rate-Limit (Twikit) - **RECOMMENDED**
- **Pros**: Much higher rate limits, bypasses API restrictions
- **Cons**: Uses browser automation (could be detected)
- **Setup**: Only requires Twitter username/password
- **Use Case**: High-volume tweeting, research purposes

To switch between methods, simply change `TWITTER_AUTH_METHOD` in your `.env` file.

## Usage

Run the script:

```sh
python main.py
```

The script will:

1. Authenticate to Twitter using the provided credentials.
2. Fetch the latest research papers from arXiv.
3. **Look up author Twitter handles** using ORCID profiles.
4. Download the PDFs and extract the text.
5. Summarize the papers using OpenAI's GPT model.
6. Post the summaries to Twitter as a thread **with author tags**.

### Configuration Options

You can customize the script behavior:

```python
tweet_arxiv_papers(
    debug=True,              # Set to False for actual posting
    days=5,                  # Number of days to look back
    max_results=10,          # Maximum papers to process
    enable_author_tagging=True  # Enable/disable author tagging
)
```

### Author Tagging

The script implements an automatic author tagging feature:

1. **arXiv → ORCID**: Searches ORCID database for each paper author
2. **ORCID → Twitter**: Extracts Twitter handles from ORCID profiles  
3. **Auto-tag**: Adds author tags to tweets when handles are found

For more details, see [AUTHOR_TAGGING.md](AUTHOR_TAGGING.md).

### Customizing Prompts

The system uses modular prompts that can be easily customized:

```python
from prompts import PromptTemplates

# Different summary styles
technical_prompt = PromptTemplates.get_summary_prompt(text, style="technical")
simple_prompt = PromptTemplates.get_summary_prompt(text, style="simple")

# Custom thread messages
ai_opener = PromptTemplates.get_thread_opener(topic="AI")
custom_closer = PromptTemplates.get_thread_closer("Thanks for reading!")
```

For more details, see [PROMPTS.md](PROMPTS.md).

## File Structure

```
├── main.py                     # Main application logic
├── prompts.py                  # Prompts and message templates
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── AUTHOR_TAGGING.md          # Author tagging documentation
├── PROMPTS.md                 # Prompts customization guide
├── .env                       # Environment variables (create this)
├── arxiv_papers/              # Downloaded PDFs (auto-created)
├── arxiv_images/              # Extracted images (auto-created)
└── test_*.py                  # Test scripts
```

## Key Files

### main.py
The main script that performs the following tasks:
- Authenticates to Twitter.
- Fetches papers from arXiv.
- Looks up author Twitter handles.
- Downloads and processes PDFs.
- Generates summaries and posts to Twitter.

### prompts.py
Contains all text templates and prompts:
- Summary generation prompts
- Twitter thread templates
- Status and error messages
- Customizable prompt styles
- Posts the summaries to Twitter.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
