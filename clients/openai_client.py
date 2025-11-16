from langchain_openai import ChatOpenAI
import os

# Global variable to hold the LLM instance
_llm = None

def get_llm():
    """
    Returns a ChatOpenAI instance, initializing it only if it hasn't been already.
    This is "lazy initialization".
    """
    global _llm
    if _llm is None:
        print("Initializing LLM client for the first time...")
        base_url = os.getenv("LLAMA_SERVER_URL")
        if not base_url:
            raise ValueError("LLAMA_SERVER_URL environment variable not set.")

        # Ensure the base_url has the /v1 suffix, as expected by the ChatOpenAI client
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip('/') + "/v1"

        _llm = ChatOpenAI(
            base_url=base_url,
            api_key="no_api_key",
            temperature=0,
            streaming=True,
        )
    return _llm

def get_completion(prompt: str) -> str:
    """
    A simple helper for getting a completion string from the LLM.
    """
    llm = get_llm()
    return llm.invoke(prompt).content
