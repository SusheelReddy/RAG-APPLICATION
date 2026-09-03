import requests
import numpy as np
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("GOOGLE_API_KEY")


def get_embeddings(text, model="gemini-embedding-001"):

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    payload = {
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    # Check API response
    print("Status code:", response.status_code)
    print("Response:", response.text[:500])

    # Raise error if API request failed
    response.raise_for_status()

    # Extract embedding
    embedding = response.json()["embedding"]["values"]

    return np.array(embedding)

