"""
OpenAI client module
Handles communication with OpenAI API for text generation
"""

import time

import requests

from config import OPENAI_BEARER_TOKEN


def get_text(prompt, model="gpt-4o-mini"):
    """
    Get text response from OpenAI API

    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use for generation

    Returns:
        str: Generated text response
    """
    try:
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENAI_BEARER_TOKEN,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )
        response = response.json()["choices"][0]["message"]["content"].replace("\n", "")
        time.sleep(3)  # Wait for 3 seconds to avoid rate limiting
    except Exception as e:
        response = ""
        print(f"An error occurred: {e}")

    return response
