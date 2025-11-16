import requests
import json

from rag import config

def get_completion(prompt: str, 
                   temperature: float = 0.8, 
                   max_tokens: int =512)-> str | None:
    """
    Generic function to get a completion from the LLM server.
    """

    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stop": ["\n"]
    }

    try:
        completion_url = f"{config.LLAMA_SERVER_URL}/completion"
        response = requests.post(completion_url,
                                 headers=headers,
                                 data=json.dumps(data))
        response.raise_for_status()

        response_data = response.json()
        content = response_data.get("content", "").strip()
        return content
    
    except requests.exceptions.RequestException as e:
        print(f"LLM request failed: {e}")
        return None
