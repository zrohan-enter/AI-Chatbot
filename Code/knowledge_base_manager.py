import json
import os
from typing import Dict, Optional

class KnowledgeBaseManager:
    """Manages loading and saving the knowledge base from a JSON file."""
   
    def __init__(self, file_path: str = "knowledge_base.json"):
        self.file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self) -> Dict:
        """Load the knowledge base from a JSON file."""
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading knowledge base: {e}")
            return {"questions": []}

    def save_knowledge_base(self):
        """Save the knowledge base to a JSON file."""
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.knowledge_base, file, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save knowledge base: {e}")

    def get_answer_for_question(self, question: str) -> tuple[Optional[str], Optional[str]]:
        """Retrieve an answer for a given question from the knowledge base."""
        for q in self.knowledge_base["questions"]:
            if q["question"].lower() == question.lower():
                return q["answer"], q["question"]
        return None, None

    def add_Youtube(self, question: str, answer: str):
        """Add a new question-answer pair to the knowledge base."""
        if not question.strip() or not answer.strip():
            raise ValueError("Question and answer cannot be empty")
        self.knowledge_base["questions"].append({"question": question, "answer": answer})
        self.save_knowledge_base()

    def reset_knowledge_base(self):
        """Reset the knowledge base to an empty state."""
        self.knowledge_base = {"questions": []}
        self.save_knowledge_base()