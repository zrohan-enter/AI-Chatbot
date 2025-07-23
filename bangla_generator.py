import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings

# Suppress specific UserWarnings from transformers library
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 4.38. Use `force_download=True` instead.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Some weights of the model checkpoint were not used when initializing",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The `padding_side` argument has been deprecated and will be removed in v4.32. Please set `tokenizer.padding_side` instead.",
    category=UserWarning
)
# Suppress warnings about T5 models from AutoModelForSeq2SeqLM if it's still in the environment
warnings.filterwarnings(
    "ignore",
    message="Some weights of T5ForConditionalGeneration were not initialized from the model checkpoint at",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="An instance of `_T5ForConditionalGeneration` was created and is being loaded from a TF checkpoint",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Could not find `pad_token_id` in tokenizer's config. Using `eos_token_id` instead.",
    category=UserWarning
)

class BanglaGenerator:
    """
    A class to handle Bengali text generation using the shahidul034/Bangla_text_generation model.
    This model is a GPT-style Causal Language Model.
    """
    def __init__(self, model_name: str = "shahidul034/Bangla_text_generation"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        print(f"Bot: Loading Bangla Text Generation model: {self.model_name} (approx 0.5GB)...")
        self._load_model()
        if self.model and self.tokenizer:
            print("Bot: Bangla Text Generation model loaded successfully.")
        else:
            print("Bot: Failed to load Bangla Text Generation model. Bangla generation will not be available.")

    def _load_model(self):
        """
        Loads the GPT-style causal language model and tokenizer.
        """
        try:
            # Use AutoTokenizer and AutoModelForCausalLM for GPT-style models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval() # Set model to evaluation mode for inference

            # GPT-2 tokenizers often lack a pad_token, use eos_token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Set padding side to left for generation tasks with decoder-only models
            self.tokenizer.padding_side = "left"

            # Ensure model is on CPU if CUDA is not available or not desired
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        except Exception as e:
            print(f"Error loading Bangla Text Generation model {self.model_name}: {e}")
            print("Please ensure internet/connectivity and that all necessary libraries are installed.")
            self.model = None
            self.tokenizer = None

    async def generate_bangla_text(self, prompt: str, max_new_tokens: int = 50, num_return_sequences: int = 1) -> str:
        """
        Generates Bengali text based on a given prompt.
        """
        if not self.model or not self.tokenizer:
            return "দুঃখিত, বাংলা টেক্সট জেনারেশন মডেল লোড করা হয়নি।" # Sorry, Bangla text generation model not loaded.

        print("Bot: Generating Bangla text...")
        try:
            input_text = prompt

            # Tokenize the input. max_length set to avoid issues, truncation to handle long prompts.
            # Ensure tokens are on the correct device (CPU)
            # ... (inside generate_bangla_text method) ...
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)

            print(f"Debug: Input IDs shape: {inputs['input_ids'].shape}")
            print(f"Debug: Attention Mask shape: {inputs['attention_mask'].shape}")
            print(f"Debug: Model max position embeddings: {self.model.config.max_position_embeddings}")
            print(f"Debug: Tokenizer model max length: {self.tokenizer.model_max_length}")

            generated_ids = await asyncio.to_thread(
                self.model.generate,
                **inputs, # Pass input_ids and attention_mask
                max_new_tokens=max_new_tokens, # Use max_new_tokens
                num_return_sequences=num_return_sequences,
                do_sample=True,             # Enable sampling for more creative output
                top_k=50,                   # Consider top 50 most likely tokens
                top_p=0.95,                 # Nucleus sampling: pick from top tokens that sum up to 95% probability
                temperature=0.7,            # Controls randomness (lower means more predictable)
                pad_token_id=self.tokenizer.pad_token_id, # Explicitly pass pad_token_id
                eos_token_id=self.tokenizer.eos_token_id, # Explicitly pass eos_token_id for robust stopping
                no_repeat_ngram_size=2      # Avoid repeating 2-grams
            )
            # Decode the generated tokens back to text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Post-process: remove the input prompt from the generated text
            # Be careful with this, as tokenization might slightly alter spacing/characters
            # A more robust check might involve comparing token IDs or checking after stripping.
            # For simplicity, if it strictly starts with the prompt, remove it.
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()

            # Ensure it ends with proper punctuation if it seems like a complete sentence
            if generated_text and not generated_text.strip().endswith(("!", "?", "।", ".")):
                generated_text += "।"

            return generated_text.strip()
        except Exception as e:
            print(f"Error during Bangla text generation: {e}")
            # Consider if the error is due to prompt being too long AFTER truncation
            # Or if it's truly an internal model limit
            return "দুঃখিত, বাংলা টেক্সট জেনারেশন তৈরি করতে সমস্যা হচ্ছে।" # Sorry, there's a problem generating Bangla text.

# Example usage (for testing this module directly)
if __name__ == '__main__':
    async def test_generator():
        generator = BanglaGenerator()
        if generator.model:
            print("\nTesting Bangla Text Generation:")
            prompt = "বাংলাদেশের রাজধানী"
            generated_text = await generator.generate_bangla_text(prompt, max_new_tokens=50)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")

            prompt2 = "আমি একজন ভালো"
            generated_text2 = await generator.generate_bangla_text(prompt2, max_new_tokens=30)
            print(f"\nPrompt: '{prompt2}'")
            print(f"Generated: '{generated_text2}'")
        else:
            print("Bangla Generator not loaded, cannot test.")

    asyncio.run(test_generator())