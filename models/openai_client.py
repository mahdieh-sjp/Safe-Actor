from openai import OpenAI
import random
import time
from tqdm.auto import tqdm

class OpenAIBatchClient:
    def __init__(self, model_name="gpt-5-mini", max_retries=7, base_delay=1):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = OpenAI(timeout=900.0)
        
    def __call_with_retry(self, input, persona):
        """
        Calls the OpenAI API with retry logic, handling rate limits.

        Args:
            model: The OpenAI model instance.
            prompt: The prompt to send to the model.
            max_retries: The maximum number of retry attempts.
            base_delay: The initial delay in seconds before the first retry.

        Returns:
            The response from the OpenAI API, or None if all retries fail.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.with_options(timeout=900.0).responses.create(model=self.model_name, input=input, instructions=persona,  reasoning={"effort": "none"},service_tier="flex")
                return response
            except Exception as e:
                if "429" in str(e):  # Check for rate limit error
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                elif "503" in str(e):  # Check for model overload
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"Model overloaded error. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                elif "502" in str(e):  # bad gateway error
                    delay = 30  # Fixed delay
                    print(f"Bad gateway error. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                elif "400" in str(e):
                    print("Message was flagged")
                    return e
                elif "Server disconnected" in str(e):
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"Server disconnected. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"An unexpected error occurred: {e}")
                    time.sleep(delay)
        print(f"Max retries ({self.max_retries}) reached.  Could not get a response.")
        return None

    def run_prompt(self, input, persona):
        """
        Run a batch of prompts. 
        For large runs, split the list to avoid rate limits.
        Returns: list of string responses.
        """
        resp = self.__call_with_retry(input, persona)
        print(resp)
        if "Error code: 400" in resp: return "Null"
        return resp.output_text if resp else None