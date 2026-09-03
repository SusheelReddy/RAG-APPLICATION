import requests
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("GOOGLE_API_KEY")

def generate_completion(prompt, model="gemini-3.1-flash-lite"):

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 200,
            "temperature": 0.3
        }
    }
    
    print("MODEL:", model)
    print("API KEY EXISTS:", bool(API_KEY))
    print("URL:", url)
    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    response.raise_for_status()

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]