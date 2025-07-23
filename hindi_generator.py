import os
import json
import requests
import asyncio

class HindiGenerator:
    """
    A class to generate Hindi text using the Google Gemini API.
    """
    def __init__(self, api_key: str = ""):
        # The API key will be provided by the Canvas environment at runtime or from an environment variable.
        # It's better to get it from an environment variable for security.
        self.api_key = os.getenv("GEMINI_API_KEY", api_key)
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables or passed. Hindi generation may not work.")
            print("Please set the GEMINI_API_KEY environment variable or pass it during initialization.")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    async def generate_hindi_text(self, user_input_hindi: str, max_output_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Generates a Hindi response using the Gemini LLM.

        Args:
            user_input_hindi (str): User input in Hindi.
            max_output_tokens (int): Max length of the bot's response.
            temperature (float): Controls creativity (0.0 - 1.0).

        Returns:
            str: Bot's response in Hindi, or an error message.
        """
        if not self.api_key:
            return "क्षमा करें, हिंदी जनरेशन के लिए API कुंजी उपलब्ध नहीं है।" # Sorry, API key not available for Hindi generation.

        print("Bot: हिंदी टेक्स्ट जनरेट कर रहा है... (Generating Hindi text...)") # Loading indicator

        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": user_input_hindi}]})

        payload = {
            "contents": chat_history,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            }
        }

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Make the API call to Gemini asynchronously using asyncio.to_thread
            response = await asyncio.to_thread(
                requests.post,
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            # Extract the text from the LLM's response
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Debug: LLM response structure unexpected: {result}")
                return "क्षमा करें, मुझे जवाब देने में समस्या हुई। (Sorry, I had trouble generating a response.)"

        except requests.exceptions.RequestException as e:
            print(f"Error during Hindi text generation (API request failed): {e}")
            return "क्षमा करें, मैं अभी उपलब्ध नहीं हूँ। कृपया बाद में पुनः प्रयास करें। (Sorry, I am not available right now. Please try again later.)"
        except json.JSONDecodeError as e:
            print(f"Error during Hindi text generation (Failed to decode JSON response): {e}")
            return "क्षमा करें, सर्वर से अमान्य प्रतिक्रिया मिली। (Sorry, received invalid response from server.)"
        except Exception as e:
            print(f"An unexpected error occurred during Hindi text generation: {e}")
            return "क्षमा करें, कुछ गलत हो गया। (Sorry, something went wrong.)"

# Example usage (for testing this module directly)
if __name__ == '__main__':
    # For testing, set a dummy API key or load from environment
    # os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE" # Replace with your actual key for testing
    async def test_generator():
        generator = HindiGenerator()
        if os.getenv("GEMINI_API_KEY"):
            print("\nTesting Hindi Text Generation:")
            prompt = "नमस्ते, आप कैसे हैं?"
            generated_text = await generator.generate_hindi_text(prompt, max_output_tokens=50)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")

            prompt2 = "भारत की राजधानी क्या है?"
            generated_text2 = await generator.generate_hindi_text(prompt2, max_output_tokens=30)
            print(f"\nPrompt: '{prompt2}'")
            print(f"Generated: '{generated_text2}'")
        else:
            print("GEMINI_API_KEY not set. Cannot test Hindi Generator directly.")

    asyncio.run(test_generator())