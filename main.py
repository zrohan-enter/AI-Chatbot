import argparse
import asyncio
import os
import sys

# Add current script directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from chatbot import Chatbot

def main():
    chatbot = Chatbot()

    parser = argparse.ArgumentParser(
        description="Multi-functional Chatbot CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Interactive Mode (optional)
    subparsers.add_parser("chat", help="Start interactive chat mode")

    # Math solver
    math_parser = subparsers.add_parser("math", help="Solve math expression")
    math_parser.add_argument("expression", type=str, help="Math expression to solve")

    # Rental prediction
    rent_parser = subparsers.add_parser("rent", help="Predict rental price")
    rent_parser.add_argument("--location", required=True, help="Location name")
    rent_parser.add_argument("--size", required=True, type=int, help="Size in sqft")
    rent_parser.add_argument("--position", default="front", choices=['front', 'back', 'corner'])
    rent_parser.add_argument("--floor", type=int, default=None, help="Floor number (optional)")
    rent_parser.add_argument("--rating", type=float, default=None, help="Rating (optional)")

    # Drug info
    drug_parser = subparsers.add_parser("drug", help="Drug info commands")
    drug_subparsers = drug_parser.add_subparsers(dest="drug_command", required=True)

    drug_info = drug_subparsers.add_parser("info", help="Get drug info")
    drug_info.add_argument("drug_name", type=str, help="Drug name")

    drug_side_effects = drug_subparsers.add_parser("side_effects", help="Get drug side effects")
    drug_side_effects.add_argument("drug_name", type=str, help="Drug name")
    drug_side_effects.add_argument("--symptom", type=str, default=None, help="Symptom for semantic search")

    drug_related = drug_subparsers.add_parser("related", help="Get related drugs")
    drug_related.add_argument("drug_name", type=str, help="Drug name")

    drug_classify = drug_subparsers.add_parser("classify", help="Classify drug as Rx or OTC")
    drug_classify.add_argument("drug_name", type=str, help="Drug name")

    # Translation
    translate_parser = subparsers.add_parser("translate", help="Translate text")
    translate_parser.add_argument("target_lang", choices=['en', 'bn', 'hi'], help="Target language code")
    translate_parser.add_argument("text", type=str, help="Text to translate")

    # Bangla and Hindi Generation
    bangla_gen_parser = subparsers.add_parser("bangla_gen", help="Generate Bangla text")
    bangla_gen_parser.add_argument("prompt", type=str, help="Prompt for Bangla text generation")

    hindi_gen_parser = subparsers.add_parser("hindi_gen", help="Generate Hindi text")
    hindi_gen_parser.add_argument("prompt", type=str, help="Prompt for Hindi text generation")

    args = parser.parse_args()

    async def run_command():
        if args.command == "chat":
            await chatbot.async_run()
        elif args.command == "math":
            result = await chatbot.solve_math_expression(args.expression)
            print(f"Bot: {result}")
        elif args.command == "rent":
            result = await chatbot.get_rent_prediction(
                location=args.location,
                size_sqft=args.size,
                position=args.position,
                floor=args.floor,
                rating=args.rating
            )
            print(f"Bot: {result}")
        elif args.command == "drug":
            if args.drug_command == "info":
                response = chatbot.get_drug_info(args.drug_name)
            elif args.drug_command == "side_effects":
                if args.symptom:
                    response = chatbot.get_semantic_side_effects(args.drug_name, args.symptom)
                else:
                    response = chatbot.get_side_effects(args.drug_name)
            elif args.drug_command == "related":
                response = chatbot.get_related_drugs(args.drug_name)
            elif args.drug_command == "classify":
                response = chatbot.classify_drug(args.drug_name)
            print(f"Bot: {response}")
        elif args.command == "translate":
            if args.target_lang == 'en':
                response = await chatbot.translate_to_english(args.text)
            elif args.target_lang == 'bn':
                response = await chatbot.translate_from_english(args.text, 'bn')
            elif args.target_lang == 'hi':
                response = await chatbot.translate_from_english(args.text, 'hi')
            print(f"Bot: {response}")
        elif args.command == "bangla_gen":
            response = await chatbot.generate_bangla_text(args.prompt)
            print(f"Bot: {response}")
        elif args.command == "hindi_gen":
            response = await chatbot.generate_hindi_text(args.prompt)
            print(f"Bot: {response}")
        else:
            parser.print_help()
    asyncio.run(run_command())


if __name__ == "__main__":
    main()
