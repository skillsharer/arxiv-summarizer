# arXiv Paper Summarizer and Twitter Bot

This script fetches the latest research papers from arXiv, summarizes them using OpenAI's GPT model, and posts the summaries to Twitter as a thread. The script is designed to run daily and provide a concise overview of the latest research across various fields.

## Features

- Fetches the latest research papers from arXiv.
- Downloads and extracts text from the PDFs.
- Summarizes the papers using OpenAI's GPT model.
- Posts the summaries to Twitter as a thread.

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

    ```env
    OPENAI_BEARER_TOKEN=your_openai_bearer_token
    CONSUMER_KEY=your_twitter_consumer_key
    CONSUMER_SECRET=your_twitter_consumer_secret
    ACCESS_TOKEN=your_twitter_access_token
    ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
    BEARER_TOKEN=your_twitter_bearer_token
    ```

## Usage

Run the script:

```sh
python main.py
```

The script will:

1. Authenticate to Twitter using the provided credentials.
2. Fetch the latest research papers from arXiv.
3. Download the PDFs and extract the text.
4. Summarize the papers using OpenAI's GPT model.
5. Post the summaries to Twitter as a thread.

## File Structure

```
.env
main.py
README.md
```

## 

main.py



The main script that performs the following tasks:

- Authenticates to Twitter.
- Sets the search period to the past 7 days.
- Defines the search query for arXiv.
- Creates a client instance for arXiv.
- Downloads and processes the results.
- Posts the summaries to Twitter.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
