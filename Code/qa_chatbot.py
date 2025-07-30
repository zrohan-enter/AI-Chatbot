from medicine_retriever import MedicineRetriever
from chat_history_manager import ChatHistoryManager
import os

def main():
    retriever = MedicineRetriever()

    # Adjust the path if needed to where your history file lives
    history_file_path = os.path.join("Result", "chat_history.txt")
    history_manager = ChatHistoryManager(history_file_path)

    while True:
        q = input("You: ")
        if q.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        answers = retriever.query(q)

        if not answers:
            print("No relevant information found.")
            # Save the "no answer" response also
            history_manager.save_qa(q, "No relevant information found.")

        else:
            # Prepare a summary string for saving in chat history
            summary_lines = []
            for i, a in enumerate(answers, 1):
                summary_lines.append(f"Result {i}:\n"
                                     f"Name: {a.get('Medicine Name', 'N/A')}\n"
                                     f"Composition: {a.get('Composition', 'N/A')}\n"
                                     f"Uses: {a.get('Uses', 'N/A')}\n"
                                     f"Side Effects: {a.get('Side Effects', 'N/A')}\n"
                                     f"Manufacturer: {a.get('Manufacturer', 'N/A')}\n"
                                     f"Reviews - Excellent: {a.get('Excellent Review %', 'N/A')}%, "
                                     f"Average: {a.get('Average Review %', 'N/A')}%, "
                                     f"Poor: {a.get('Poor Review %', 'N/A')}%")
            summary_text = "\n\n".join(summary_lines)

            # Print results to console
            for i, a in enumerate(answers, 1):
                print(f"\nResult {i}:")
                print(f"Name: {a.get('Medicine Name', 'N/A')}")
                print(f"Composition: {a.get('Composition', 'N/A')}")
                print(f"Uses: {a.get('Uses', 'N/A')}")
                print(f"Side Effects: {a.get('Side Effects', 'N/A')}")
                print(f"Manufacturer: {a.get('Manufacturer', 'N/A')}")
                print(f"Reviews - Excellent: {a.get('Excellent Review %', 'N/A')}%, "
                      f"Average: {a.get('Average Review %', 'N/A')}%, "
                      f"Poor: {a.get('Poor Review %', 'N/A')}%\n")

            # Save the question and the textual summary of answers
            history_manager.save_qa(q, summary_text)


if __name__ == "__main__":
    main()
