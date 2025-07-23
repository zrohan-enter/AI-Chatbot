import pandas as pd
import re
import joblib
import os
from typing import Optional, List, Dict, Union, Set
# Assuming NLPProcessor is available, which it should be in your project
from nlp_processor import NLPProcessor 


class DrugInfoProcessor:
    def __init__(self, data_path: str = 'drugs_side_effects_drugs_com.csv',
                 processed_data_path: str = 'drug_data.pkl',
                 nlp_processor: Optional[NLPProcessor] = None): # Accept NLPProcessor instance
        
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.nlp_processor = nlp_processor # Store the NLPProcessor
        self.drug_data: Dict[str, Dict[str, str]] = {}
        self.drug_names: Set[str] = set()
        self._load_or_process_data()

    def _load_or_process_data(self):
        # ... (This method remains largely the same as before) ...
        """
        Attempts to load processed data from a PKL file.
        If not found or an error occurs, processes from CSV and saves to PKL.
        """
        if os.path.exists(self.processed_data_path):
            try:
                print(f"Attempting to load processed drug data from {self.processed_data_path}...")
                loaded_data = joblib.load(self.processed_data_path)
                if isinstance(loaded_data, dict) and 'drug_data' in loaded_data and 'drug_names' in loaded_data:
                    self.drug_data = loaded_data['drug_data']
                    self.drug_names = loaded_data['drug_names']
                    print(f"Successfully loaded {len(self.drug_data)} drugs from {self.processed_data_path}")
                    return
                else:
                    print(f"Invalid format or corrupted data in {self.processed_data_path}. Re-processing from CSV.")
            except Exception as e:
                print(f"Error loading {self.processed_data_path}: {e}. Re-processing from CSV.")

        # If loading failed or file doesn't exist, process from CSV
        print(f"'{self.processed_data_path}' not found or invalid. Processing drug data from '{self.data_path}'...")
        self._process_data_from_csv()
        self._save_processed_data()

    def _process_data_from_csv(self):
        # ... (This method remains largely the same as before, ensuring strip and fillna) ...
        """Loads and processes the drug data from the CSV file into a dictionary."""
        try:
            full_data_path = os.path.join(os.path.dirname(__file__), self.data_path) if os.path.exists(os.path.join(os.path.dirname(__file__), self.data_path)) else self.data_path
            
            df = pd.read_csv(full_data_path)
            
            df['drug_name'] = df['drug_name'].astype(str).fillna('')
            df['generic_name'] = df['generic_name'].astype(str).fillna('')
            df['drug_classes'] = df['drug_classes'].astype(str).fillna('')
            df['activity'] = df['activity'].astype(str).fillna('')
            df['rx_otc'] = df['rx_otc'].astype(str).fillna('')
            df['pregnancy_category'] = df['pregnancy_category'].astype(str).fillna('')
            df['side_effects'] = df['side_effects'].astype(str).fillna('')
            df['related_drugs'] = df['related_drugs'].astype(str).fillna('')

            drug_info = {}
            for index, row in df.iterrows():
                drug_name = row['drug_name'].lower().strip()
                if drug_name:
                    drug_info[drug_name] = {
                        'generic_name': row['generic_name'].strip(),
                        'drug_classes': row['drug_classes'].strip(),
                        'activity': row['activity'].strip(),
                        'rx_otc': row['rx_otc'].strip(),
                        'pregnancy_category': row['pregnancy_category'].strip(),
                        'side_effects': row['side_effects'].strip(),
                        'related_drugs': row['related_drugs'].strip()
                    }
            self.drug_data = drug_info
            self.drug_names = {name for name in self.drug_data.keys() if name}
            print(f"Processed {len(self.drug_data)} unique drugs from '{self.data_path}'")
        except FileNotFoundError:
            print(f"Error: '{self.data_path}' not found. Please ensure the CSV file is in the correct directory.")
            self.drug_data = {}
            self.drug_names = set()
        except Exception as e:
            print(f"Error processing drug data from CSV: {e}")
            self.drug_data = {}
            self.drug_names = set()

    def _save_processed_data(self):
        # ... (This method remains largely the same as before) ...
        """Saves the processed drug data (dictionary and set) to a PKL file."""
        try:
            data_to_save = {'drug_data': self.drug_data, 'drug_names': self.drug_names}
            joblib.dump(data_to_save, self.processed_data_path)
            print(f"Processed drug data saved to '{self.processed_data_path}'")
        except Exception as e:
            print(f"Error saving processed drug data to '{self.processed_data_path}': {e}")


    # --- MODIFIED DRUG NAME FINDING METHOD ---
    def find_drug_in_query(self, query: str) -> Optional[str]:
        """
        Attempts to find a drug name within the user's query using both direct
        matching and NLP-based similarity (if NLPProcessor is available).
        """
        query_lower = query.lower()

        # 1. First, try direct keyword matching (fastest)
        found_drug_direct = None
        max_len_match_direct = 0
        for drug in sorted(list(self.drug_names), key=len, reverse=True):
            pattern = r'\b' + re.escape(drug) + r'\b'
            if re.search(pattern, query_lower):
                if len(drug) > max_len_match_direct:
                    found_drug_direct = drug
                    max_len_match_direct = len(drug)
        if found_drug_direct:
            print(f"Debug: Direct match found: {found_drug_direct}")
            return found_drug_direct

        # 2. If no direct match, use NLPProcessor for semantic similarity
        if self.nlp_processor and self.drug_names:
            print("Debug: No direct match, trying NLP similarity for drug name...")
            # Convert set to list for find_best_match
            best_match, score = self.nlp_processor.find_best_match(query, list(self.drug_names))
            if best_match and score > 0.7: # A slightly lower threshold might be okay for drug names
                print(f"Debug: NLP match found: {best_match} with score {score}")
                return best_match
            print(f"Debug: No strong NLP match for drug name (best: {best_match}, score: {score})")
        
        return None

    # --- ADDED METHOD FOR SEMANTIC SIDE EFFECT RETRIEVAL ---
    def get_semantically_relevant_side_effects(self, drug_name: str, symptom_query: str) -> str:
        """
        Retrieves side effects for a drug, focusing on semantic relevance to a symptom query.
        Uses NLPProcessor to compare the query to sentences within the side effects text.
        """
        drug_name_lower = drug_name.lower()
        if drug_name_lower not in self.drug_data:
            return f"I could not find information for {drug_name}. Please check the spelling."

        full_side_effects_text = self.drug_data[drug_name_lower]['side_effects']
        if not full_side_effects_text or full_side_effects_text == 'nan':
            return f"Side effects information not available for {drug_name}."

        if not self.nlp_processor:
            return f"NLP capabilities not available to semantically search side effects for {drug_name}. Full list: {full_side_effects_text}"

        # Simple sentence splitting (can be improved with nltk.sent_tokenize)
        side_effect_sentences = [s.strip() for s in re.split(r'[.;](?=\s*[A-Z])', full_side_effects_text) if s.strip()]

        relevant_sentences = []
        for sentence in side_effect_sentences:
            # Use NLPProcessor to find similarity between symptom query and side effect sentence
            # Note: find_best_match returns a tuple (best_match_text, score) when comparing
            # a query to a list of candidates. Here, we're doing it sentence by sentence.
            # A simpler way would be to get embedding for symptom_query and each sentence, then calculate cosine.
            # Let's adapt NLPProcessor.find_best_match slightly if it can return score, or do manual cosine.
            
            # Since find_best_match is designed for a list of questions, we'll feed it one sentence at a time
            # or directly calculate cosine if NLPProcessor exposes embedding method.
            # Assuming find_best_match will return (sentence, score) for direct compare.
            
            # ALTERNATIVE: Directly get embeddings and calculate cosine similarity
            try:
                query_embedding = self.nlp_processor.nlp(symptom_query)[0][0]
                sentence_embedding = self.nlp_processor.nlp(sentence)[0][0]
                score = 1 - (self.nlp_processor.cosine(query_embedding, sentence_embedding) if hasattr(self.nlp_processor, 'cosine') else self.nlp_processor.scipy_cosine(query_embedding, sentence_embedding))
                
                # A higher threshold for semantic match
                if score > 0.7: # Adjust this threshold based on testing
                    relevant_sentences.append(sentence)
            except Exception as e:
                print(f"Error during semantic search for side effects: {e}")
                # Fallback to direct text search if NLP fails
                if any(word in sentence.lower() for word in symptom_query.lower().split()):
                    relevant_sentences.append(sentence)


        if relevant_sentences:
            return (f"For {drug_name.capitalize()} regarding '{symptom_query}'-like issues, "
                    f"you might experience: {'; '.join(relevant_sentences)}.")
        else:
            # Fallback if no strong semantic match but symptom query might be a direct keyword
            if any(word in full_side_effects_text.lower() for word in symptom_query.lower().split() if len(word) > 2):
                 return (f"For {drug_name.capitalize()}, some side effects are mentioned that might relate to "
                         f"'{symptom_query}'. The full side effects are: {full_side_effects_text[:300]}...") # Truncate long text
            return f"I did not find specific side effects related to '{symptom_query}' for {drug_name.capitalize()}. The full side effects are: {full_side_effects_text[:300]}..." # Truncate

    # ... (get_drug_info, get_side_effects, get_related_drugs methods remain the same) ...
    def get_drug_info(self, drug_name: str) -> str:
        """Returns general information about a drug."""
        drug_name_lower = drug_name.lower()
        if drug_name_lower in self.drug_data:
            info = self.drug_data[drug_name_lower]
            response = (
                f"{info['generic_name'].capitalize()} is a generic drug belonging to {info['drug_classes']}. "
                f"It has {info['activity']}% activity and is an {info['rx_otc']} drug."
            )
            if info['pregnancy_category'] and info['pregnancy_category'] != 'nan':
                response += f" It is categorized as Pregnancy Category {info['pregnancy_category']}."
            return response
        return f"I could not find information for {drug_name}. Please check the spelling."

    def get_side_effects(self, drug_name: str) -> str:
        """Returns the side effects of a drug."""
        drug_name_lower = drug_name.lower()
        if drug_name_lower in self.drug_data:
            side_effects = self.drug_data[drug_name_lower]['side_effects']
            if side_effects and side_effects != 'nan':
                side_effects = side_effects.strip().strip('"')
                return f"Common side effects of {drug_name.capitalize()} may include: {side_effects}."
            return f"Side effects information not available for {drug_name}."
        return f"I could not find information for {drug_name}. Please check the spelling."

    def get_related_drugs(self, drug_name: str) -> str:
        """Returns drugs related to a given drug."""
        drug_name_lower = drug_name.lower()
        if drug_name_lower in self.drug_data:
            related_drugs_str = self.drug_data[drug_name_lower]['related_drugs']
            if related_drugs_str and related_drugs_str != 'nan':
                drug_names_only = [
                    re.split(r':\s*https?://', item)[0].strip()
                    for item in related_drugs_str.split('|') if re.split(r':\s*https?://', item)[0].strip()
                ]
                if drug_names_only:
                    return f"Yes, some drugs related to {drug_name.capitalize()} include: {', '.join(drug_names_only)}."
            return f"No related drugs information available for {drug_name}."
        return f"I could not find information for {drug_name}. Please check the spelling."