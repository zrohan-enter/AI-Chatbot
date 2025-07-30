import os


class ChatHistoryManager:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.isfile(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("")
        self.qa_dict = {}
        self.reload_history()  # Initialize by loading current file content

    def reload_history(self):
        """
        Reloads the QA history dictionary from the text file.
        Call when you want to reflect external changes or refresh memory.
        """
        qa = {}
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Each QA entry: 3 lines (Q, A, blank)
            for i in range(0, len(lines), 3):
                if i + 1 < len(lines):
                    q_line = lines[i].strip()
                    a_line = lines[i + 1].strip()
                    if q_line.startswith("Q: ") and a_line.startswith("A: "):
                        question = q_line[3:]
                        answer = a_line[3:]
                        qa[question] = answer
        except Exception as e:
            print(f"Error loading history file: {e}")
        self.qa_dict = qa

    def get_answer(self, question):
        """
        Reload history automatically before getting the answer,
        so you always get the latest saved info.
        """
        self.reload_history()
        return self.qa_dict.get(question)

    def save_qa(self, question, answer):
        """
        Append a Q&A pair to the history file and update in-memory dictionary immediately.
        """
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(f"Q: {question}\n")
                f.write(f"A: {answer}\n\n")  # blank line as separator
            # Update in-memory dict to avoid reloading immediately
            self.qa_dict[question] = answer
        except Exception as e:
            print(f"Error saving to history file: {e}")
    def add_interaction(self, question, answer):
        self.save_qa(question, answer)
