
from google import genai
import random
import time
from tqdm.auto import tqdm

class GeminiBatchClient:
    def __init__(self, model_name="gemini-3-pro-preview", max_retries=7, base_delay=1):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = genai.Client()

    def __call_with_retry(self, prompt):
        """
        Calls the Gemini API with retry logic, handling rate limits.

        Args:
            model: The Gemini model instance.
            prompt: The prompt to send to the model.
            max_retries: The maximum number of retry attempts.
            base_delay: The initial delay in seconds before the first retry.

        Returns:
            The response from the Gemini API, or None if all retries fail.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.interactions.create(model=self.model_name, input=prompt, service_tier="flex")
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

    def run_prompt(self, prompt):
        """
        Run a batch of prompts. 
        For large runs, split the list to avoid rate limits.
        Returns: list of string responses.
        """
        
        resp = self.__call_with_retry(prompt)
        print(resp)
        return resp.output_text if resp else None