from typing import List, Dict, Optional

class TrainDataManager:
    """Manages loading question-answer pairs from a text file (Train.txt)."""
   
    def __init__(self, file_path: str = "Train.txt"):
        self.file_path = file_path
        self.train_data = self.load_train_file()

    def load_train_file(self) -> List[Dict[str, str]]:
        """Load question-answer pairs from Train.txt."""
        train_data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                i = 0
                while i < len(lines):
                    if lines[i].startswith("Question: ") and i + 1 < len(lines) and lines[i + 1].startswith("Answer: "):
                        question = lines[i][len("Question: "):].strip()
                        answer = lines[i + 1][len("Answer: "):].strip()
                        train_data.append({"question": question, "answer": answer})
                        i += 2
                    else:
                        i += 1
                return train_data
        except FileNotFoundError:
            print(f"Bot: Warning: {self.file_path} not found. Skipping local text file search.")
            return []
        except Exception as e:
            print(f"Bot: Error reading {self.file_path}: {e}")
            return []

    def find_answer_in_train(self, question: str, nlp_processor) -> Optional[tuple[str, str]]:
        """Find an answer in the train data, using NLP for similarity if needed."""
        for q in self.train_data:
            if q["question"].lower() == question.lower():
                return q["answer"], q["question"]
        questions = [q["question"] for q in self.train_data]
        best_match = nlp_processor.find_best_match(question, questions)
        if best_match:
            for q in self.train_data:
                if q["question"].lower() == best_match.lower():
                    return q["answer"], best_match
        return None, None