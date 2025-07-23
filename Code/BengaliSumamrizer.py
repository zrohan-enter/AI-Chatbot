# Save this as bangla_summarizer.py in your project directory

import asyncio
import torch
from transformers import pipeline, AutoTokenizer
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
warnings.filterwarnings(
    "ignore",
    message="Some weights of T5ForConditionalGeneration were not initialized from the model checkpoint at",
    category=UserWarning
)

class BanglaSummarizer:
    """
    A class to handle Bengali text summarization using the kawsarahmd/bnT5-64k_xlsum_bangla_v2 model.
    Uses Hugging Face pipeline for simplified loading and inference.
    """
    def __init__(self, model_name: str = "kawsarahmd/bnT5-64k_xlsum_bangla_v2"):
        self.model_name = model_name
        self.summarizer_pipeline = None
        self.tokenizer = None

        print(f"Bot: Loading Bangla Summarization model: {self.model_name} (first time might download ~350MB)...")
        self._load_model()
        if self.summarizer_pipeline:
            print("Bot: Bangla Summarization model loaded successfully.")
        else:
            print("Bot: Failed to load Bangla Summarization model. Summarization will not be available.")

    def _load_model(self):
        """
        Loads the summarization pipeline using AutoModelForSeq2SeqLM and AutoTokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device="cpu", # Explicitly set to CPU
                # NEW: Add from_flax=True to load Flax weights
                model_kwargs={"from_flax": True} 
            )
        except Exception as e:
            #print(f"Error loading Bangla summarization model {self.model_name}: {e}")
            #print("Please ensure internet/connectivity and that all necessary libraries (like sentencepiece) are installed.")
            self.summarizer_pipeline = None
            self.tokenizer = None

    async def summarize_bangla_text(self, text: str) -> str:
        """
        Summarizes a given Bengali text using the loaded summarization pipeline.
        """
        if not self.summarizer_pipeline:
            return "দুঃখিত, সারসংক্ষেপ মডেল লোড করা হয়নি।" # Sorry, summarization model not loaded.

        print("Bot: Generating summary (this may take a moment on CPU)...")
        try:
            summary_results = await asyncio.to_thread(
                self.summarizer_pipeline,
                text,
                min_length=10,
                max_length=64,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            summarized_text = summary_results[0]['summary_text']

            if summarized_text and not summarized_text.strip().endswith(("!", "?", "।", ".")):
                summarized_text += "।"

            return summarized_text.strip()
        except Exception as e:
            print(f"Error during Bangla text summarization: {e}")
            return "দুঃখিত, বাংলা পাঠের সারসংক্ষেপ তৈরি করতে সমস্যা হচ্ছে।"