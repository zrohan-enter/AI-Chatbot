from typing import List, Optional, Tuple
from transformers import pipeline
from scipy.spatial.distance import cosine

class NLPProcessor:
    def __init__(self):
        self.nlp = pipeline("feature-extraction", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")
        self.cosine = cosine

    def get_embedding(self, text: str) -> List[float]:
        return self.nlp(text)[0][0].tolist()

    def find_best_match(self, user_question: str, questions: List[str]) -> Tuple[Optional[str], float]:
        try:
            user_embedding = self.get_embedding(user_question)
            best_score = 0.0
            best_question = None
            for question in questions:
                q_embedding = self.get_embedding(question)
                score = 1 - self.cosine(user_embedding, q_embedding)
                if score > best_score and score > 0.85:
                    best_score = score
                    best_question = question
            return best_question, best_score
        except Exception:
            return None, 0.0
