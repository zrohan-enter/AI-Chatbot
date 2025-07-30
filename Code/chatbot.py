import os
import sys
import asyncio
from dotenv import load_dotenv
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Add current dir for imports (adjust if needed)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import your modules (make sure these files exist and are correct)
from rental_predictor import RentalPredictor
from knowledge_base_manager import KnowledgeBaseManager
from train_data_manager import TrainDataManager
from nlp_processor import NLPProcessor
from web_searcher import WebSearcher
from math_solver import MathSolver
from translator import Translator
from bangla_generator import BanglaGenerator
from hindi_generator import HindiGenerator
from drug_info_processor import DrugInfoProcessor
from drug_classifier import DrugClassifier

load_dotenv()

class Chatbot:
    def __init__(self):
        print("Chatbot: Initializing core components (task-specific features only)...")

        self.knowledge_base_manager = KnowledgeBaseManager()
        self.train_data_manager = TrainDataManager()
        self.nlp_processor = NLPProcessor()
        self.drug_info_processor = DrugInfoProcessor(nlp_processor=self.nlp_processor)
        self.drug_classifier = DrugClassifier()
        self.web_searcher = WebSearcher()
        self.math_solver = MathSolver()
        self.translator = Translator()
        self.bangla_generator = BanglaGenerator()
        self.hindi_generator = HindiGenerator()

        self.rental_predictor = None
        try:
            print("Chatbot: Attempting to load RentalPredictor from model files...")
            self.rental_predictor = RentalPredictor(model_path='rental_predictor_model.pkl', columns_path='original_X_columns.pkl')
            print("Chatbot: RentalPredictor loaded from model files.")
        except Exception as e:
            print(f"Chatbot: Warning: RentalPredictor could not be initialized from model files: {e}")
            try:
                print("Chatbot: Attempting to initialize RentalPredictor from raw data...")
                self.rental_predictor = RentalPredictor(data_path='rental_data.csv')
                print("Chatbot: RentalPredictor initialized from data path.")
            except Exception as e:
                print(f"Chatbot: Error: RentalPredictor could not be initialized from data either: {e}")
                self.rental_predictor = None
                print("Chatbot: Rental predictor unavailable.")

    async def async_run(self, history_manager=None):
        """
        Basic interactive mode (placeholder).
        Can pass a ChatHistoryManager instance to save history automatically.
        """
        print("Type 'exit' or 'quit' to stop.")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            if not user_input:
                continue
            # For now, this mode only prints a placeholder message until chat functionality is added.
            print("Bot: Please use command-specific queries via CLI or implement handlers for this mode.")
            # If you want to add real response generation, insert it here and save QA:
            # Example:
            # response = await self.get_response(user_input)
            # print(f"Bot: {response}")
            # if history_manager:
            #     history_manager.save_qa(user_input, response)

    # Task-specific async methods
    async def get_rent_prediction(self, location: str, size_sqft: int, position='front', floor=None, rating=None) -> str:
        if not self.rental_predictor:
            return "Rental predictor unavailable."
        try:
            rent = self.rental_predictor.predict_rental_price(location, size_sqft, position, floor, rating)
            return f"Estimated rent for {size_sqft} sqft at {location} ({position}) is {rent:.2f} BDT."
        except Exception as e:
            return f"Error in rental prediction: {e}"

    async def solve_math_expression(self, expression: str) -> str:
        return self.math_solver.solve_math(expression)

    def get_drug_info(self, drug_name: str) -> str:
        return self.drug_info_processor.get_drug_info(drug_name)

    def get_side_effects(self, drug_name: str) -> str:
        return self.drug_info_processor.get_side_effects(drug_name)

    def get_semantic_side_effects(self, drug_name: str, symptom: str) -> str:
        return self.drug_info_processor.get_semantically_relevant_side_effects(drug_name, symptom)

    def get_related_drugs(self, drug_name: str) -> str:
        return self.drug_info_processor.get_related_drugs(drug_name)

    def classify_drug(self, drug_name: str) -> str:
        drug_info = self.drug_info_processor.drug_data.get(drug_name.lower())
        if not drug_info:
            return f"No data available to classify {drug_name}."
        features = {
            'activity': drug_info.get('activity', ''),
            'drug_classes': drug_info.get('drug_classes', '')
        }
        prediction = self.drug_classifier.predict_rx_otc(features)
        if prediction:
            return f"{drug_name.capitalize()} is predicted to be an **{prediction}** drug."
        return "Could not classify drug."

    async def translate_to_english(self, text: str) -> str:
        return await self.translator.translate_bengali_to_english(text)

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        return await self.translator.translate_from_english(text, target_lang)

    async def generate_bangla_text(self, prompt: str) -> str:
        return await self.bangla_generator.generate_bangla_text(prompt)

    async def generate_hindi_text(self, prompt: str) -> str:
        return await self.hindi_generator.generate_hindi_text(prompt)
