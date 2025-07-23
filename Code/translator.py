# translator.py
from deep_translator import GoogleTranslator
import asyncio

class Translator:
    """
    A class to handle language translation using deep_translator,
    leveraging online Google Translate's free web interface.
    """

    def __init__(self):
        # Initialize GoogleTranslator instances for English↔Bengali
        self.en_to_bn = GoogleTranslator(source='en', target='bn')
        self.bn_to_en = GoogleTranslator(source='bn', target='en')
        print("Google Translate initialized successfully using deep_translator.")

    async def translate_english_to_bengali(self, text: str) -> str:
        """
        Translates an English sentence into Bengali.
        """
        try:
            result = await asyncio.to_thread(self.en_to_bn.translate, text)
            return result
        except Exception as e:
            print(f"Error during English to Bengali translation: {e}")
            return "দুঃখিত, ইংরেজি থেকে বাংলা অনুবাদ করতে সমস্যা হচ্ছে।"

    async def translate_bengali_to_english(self, text: str) -> str:
        """
        Translates a Bengali sentence into English.
        """
        try:
            result = await asyncio.to_thread(self.bn_to_en.translate, text)
            return result
        except Exception as e:
            print(f"Error during Bengali to English translation: {e}")
            return "Sorry, there's a problem translating Bengali to English."

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translates English text to any supported target language.

        target_lang: e.g., 'bn' for Bengali, 'hi' for Hindi, etc.
        """
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            result = await asyncio.to_thread(translator.translate, text)
            return result
        except Exception as e:
            print(f"Error during English to {target_lang} translation: {e}")
            fallback = {
                'bn': "দুঃখিত, ইংরেজি থেকে বাংলা অনুবাদ করতে সমস্যা হচ্ছে।",
                'hi': "माफ़ करें, अंग्रेज़ी से हिंदी अनुवाद करने में समस्या हो रही है।"
            }
            return fallback.get(target_lang, f"Sorry, translation to {target_lang} failed.")
