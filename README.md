# arXiv Paper Summarizer and Twitter Bot

This script fetches the latest research papers from arXiv, summarizes them using OpenAI's GPT model, and posts the summaries to Twitter as a thread. The script is designed to run daily and provide a concise overview of the latest research across various fields.

## Features

- Fetches the latest research papers from arXiv.
- Downloads and extracts text from the PDFs.
- Summarizes the papers using OpenAI's GPT model.
- Posts the summaries to Twitter as a thread.
- **Author Tagging**: Automatically tags paper authors on Twitter using arXiv → ORCID → Twitter pipeline.
- **Modular Architecture**: Clean, maintainable codebase with separated modules for each functionality.
- **Modern Package Management**: Uses `uv` for fast dependency management and development tools.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip for package management

## Setup

1. **Clone the repository:**

    ```sh
    git clone git@github.com:skillsharer/arxiv-summarizer.git
    cd arxiv-summarizer
    ```

2. **Install dependencies using uv (recommended):**

    ```sh
    # Install uv if you don't have it
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Install all dependencies
    uv sync
    
    # Or install with dev dependencies for development
    uv sync --dev
    ```

    **Alternative: Traditional setup with pip:**

    ```sh
    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3. **Create a .env file in the root directory and add your API keys:**

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

### Running the Application

**With uv (recommended):**
```sh
uv run python src/main.py
```

**With traditional Python:**
```sh
python src/main.py
```

### Development Commands

**With uv:**
```sh
# Install all dependencies
uv sync

# Install with dev dependencies  
uv sync --dev

# Run development tools
uv run black .        # Format code
uv run isort .        # Sort imports  
uv run mypy src/      # Type checking
uv run pytest        # Run tests

# Run your application
uv run python src/main.py
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
from src.main import tweet_arxiv_papers

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
from src.prompts import PromptTemplates

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
├── src/                       # Source code modules
│   ├── main.py               # Main application logic
│   ├── config.py             # Configuration and environment variables
│   ├── twitter_auth.py       # Twitter authentication handling
│   ├── twitter_client.py     # Twitter posting functionality
│   ├── author_lookup.py      # Author Twitter handle discovery
│   ├── pdf_processor.py      # PDF processing and image extraction
│   ├── paper_search.py       # ArXiv paper search and AI scoring
│   ├── openai_client.py      # OpenAI API interactions
│   ├── prompts.py            # Prompts and message templates
│   └── utils.py              # Utility functions
├── pyproject.toml            # Modern Python project configuration
├── uv.lock                   # Locked dependency versions
├── requirements.txt          # Legacy dependency list
├── README.md                 # This file
├── AUTHOR_TAGGING.md         # Author tagging documentation
├── PROMPTS.md                # Prompts customization guide
├── .env                      # Environment variables (create this)
├── arxiv_papers/             # Downloaded PDFs (auto-created)
├── arxiv_images/             # Extracted images (auto-created)
└── test_*.py                 # Test scripts
```

## Key Modules

### src/main.py
The main orchestration script that coordinates all functionality:
- Integrates all modules
- Provides the main `tweet_arxiv_papers()` function
- Handles the complete workflow from search to posting

### src/config.py
Central configuration management:
- Environment variable handling
- Global client configuration
- Authentication method selection

### src/twitter_auth.py
Twitter authentication handling:
- Supports both twikit and tweepy methods
- Session management and validation
- Fallback authentication guidance

### src/twitter_client.py
Twitter posting functionality:
- Unified interface for both auth methods
- Tweet formatting and media upload
- Thread creation and author tagging

### src/author_lookup.py
Author Twitter handle discovery:
- ORCID profile integration
- Semantic Scholar API integration
- Enhanced lookup algorithms with rate limiting

### src/pdf_processor.py
PDF processing and image extraction:
- Text extraction from academic papers
- Image quality filtering
- GIF creation for Twitter media

### src/paper_search.py
ArXiv paper search and AI scoring:
- Multi-topic paper discovery
- AI-powered engagement scoring
- Diversity optimization for paper selection

### src/openai_client.py
OpenAI API interactions:
- GPT model integration
- Summary generation
- Viral tweet creation

### src/prompts.py
Prompts and message templates:
- Summary generation prompts
- Twitter thread templates
- Status and error messages
- Customizable prompt styles

### src/utils.py
Utility functions:
- File management
- Directory handling
- Common helper functions

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
