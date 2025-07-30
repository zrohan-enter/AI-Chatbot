import argparse
import asyncio
import os
import sys

# Add current dir for imports (adjust if needed)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import QwenChatbot for the chat command
from qwen_chatbot import QwenChatbot


from chat_history_manager import ChatHistoryManager

def main():
    parser = argparse.ArgumentParser(description="Run the AI Chatbot with different commands.")
    parser.add_argument(
        "command",
        choices=[
            "chat", "history", "clear_history", "predict_rent", "solve_math",
            "get_drug_info", "get_side_effects", "get_semantic_side_effects",
            "get_related_drugs", "classify_drug", "translate_to_english",
            "translate_from_english", "generate_bangla", "generate_hindi"
        ],
        help="Command to execute."
    )
    parser.add_argument("--query", type=str, help="Query for the command (e.g., math expression, drug name, text to translate).")
    parser.add_argument("--location", type=str, help="Location for rent prediction.")
    parser.add_argument("--size", type=int, help="Size in sqft for rent prediction.")
    parser.add_argument("--position", type=str, default="front", help="Position for rent prediction (e.g., 'front', 'back').")
    parser.add_argument("--floor", type=int, help="Floor number for rent prediction.")
    parser.add_argument("--rating", type=float, help="Rating for rent prediction.")
    parser.add_argument("--target_lang", type=str, help="Target language for translation (e.g., 'bn' for Bengali, 'hi' for Hindi).")
    parser.add_argument("--symptom", type=str, help="Symptom for semantic side effects search.")

    args = parser.parse_args()

    # Initialize the original Chatbot instance if available for non-chat commands
    try:
        from chatbot import Chatbot
        chatbot = Chatbot()
    except ImportError:
        print("Warning: 'chatbot.py' or 'Chatbot' class not found. Some commands may not work.")
        chatbot = None

    # Initialize chat history manager
    history_file_path = r"D:\MY Ai\CSE 299\Chatbot\Result\chat_history.txt"
    history_manager = ChatHistoryManager(history_file_path)

    async def run_command():
        try:
            # Load history if supported
            history = history_manager.load_history() if hasattr(history_manager, "load_history") else None
            if history:
                print(f"Bot: Loaded chat history from {history_file_path}")

            if args.command == "chat":
                print("Initializing Qwen-powered interactive chat mode. Please wait for model loading...")
                qwen_chatbot_instance = QwenChatbot()

                print("Entering interactive chat mode. Type 'exit' to leave.")
                while True:
                    user_input = input("You: ").strip()
                    if user_input.lower() == "exit":
                        print("Bot: Goodbye!")
                        break

                    saved_answer = history_manager.get_answer(user_input)
                    if saved_answer:
                        print(f"Bot (from history): {saved_answer}")
                    else:
                        response = await asyncio.to_thread(qwen_chatbot_instance.generate_response, user_input)
                        print(f"Bot: {response}")
                        history_manager.save_qa(user_input, response)

            elif args.command == "history":
                history_manager.display_history()

            elif args.command == "clear_history":
                history_manager.clear_history()
                print("Bot: Chat history cleared.")

            elif args.command == "predict_rent":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.location or not args.size:
                    print("Error: For 'predict_rent', --location and --size are required.")
                else:
                    response = await chatbot.get_rent_prediction(args.location, args.size, args.position, args.floor, args.rating)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(
                        f"Predict Rent: Location={args.location}, Size={args.size}, Position={args.position}, Floor={args.floor}, Rating={args.rating}", response)

            elif args.command == "solve_math":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'solve_math', --query (the expression) is required.")
                else:
                    response = await chatbot.solve_math_expression(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Solve Math: {args.query}", response)

            elif args.command == "get_drug_info":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'get_drug_info', --query (the drug name) is required.")
                else:
                    response = chatbot.get_drug_info(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Get Drug Info: {args.query}", response)

            elif args.command == "get_side_effects":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'get_side_effects', --query (the drug name) is required.")
                else:
                    response = chatbot.get_side_effects(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Get Side Effects: {args.query}", response)

            elif args.command == "get_semantic_side_effects":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query or not args.symptom:
                    print("Error: For 'get_semantic_side_effects', --query (drug name) and --symptom are required.")
                else:
                    response = chatbot.get_semantic_side_effects(args.query, args.symptom)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Get Semantic Side Effects: Drug={args.query}, Symptom={args.symptom}", response)

            elif args.command == "get_related_drugs":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'get_related_drugs', --query (the drug name) is required.")
                else:
                    response = chatbot.get_related_drugs(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Get Related Drugs: {args.query}", response)

            elif args.command == "classify_drug":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'classify_drug', --query (the drug name) is required.")
                else:
                    response = chatbot.classify_drug(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Classify Drug: {args.query}", response)

            elif args.command == "translate_to_english":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'translate_to_english', --query (text) is required.")
                else:
                    response = await chatbot.translate_to_english(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Translate to English: {args.query}", response)

            elif args.command == "translate_from_english":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query or not args.target_lang:
                    print("Error: For 'translate_from_english', --query (text) and --target_lang are required.")
                else:
                    response = await chatbot.translate_from_english(args.query, args.target_lang)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Translate from English: {args.query} to {args.target_lang}", response)

            elif args.command == "generate_bangla":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'generate_bangla', --query (prompt) is required.")
                else:
                    response = await chatbot.generate_bangla_text(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Generate Bangla: {args.query}", response)

            elif args.command == "generate_hindi":
                if not chatbot:
                    print("Error: 'Chatbot' instance not available for this command.")
                elif not args.query:
                    print("Error: For 'generate_hindi', --query (prompt) is required.")
                else:
                    response = await chatbot.generate_hindi_text(args.query)
                    print(f"Bot: {response}")
                    history_manager.add_interaction(f"Generate Hindi: {args.query}", response)

            else:
                print("Unknown command.")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_command())

if __name__ == "__main__":
    main()
