import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPTChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        print(f"Loading model {model_name} for CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Explicitly move model to CPU
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        self.chat_history_ids = None
        print("Model loaded! Start chatting (type 'exit' to quit).")

    def get_response(self, user_input):
        print("--- Processing input ---") # Debug print
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt').to("cpu")

        if self.chat_history_ids is not None:
            # Ensure history is also on CPU
            bot_input_ids = torch.cat([self.chat_history_ids.to("cpu"), new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Create attention mask: 1 for tokens, 0 for padding, ensure on CPU
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long).to("cpu")

        print("--- Generating response ---") # Debug print
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=150,  # Adjusted for potentially faster CPU generation
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.85,
            top_k=40,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        print("--- Decoding response ---") # Debug print
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response


def main():
    chatbot = DialoGPTChatbot()
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Bot: Goodbye!")
            break
        try:
            response = chatbot.get_response(user_input)
            print("Bot:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting conversation. Try again.")
            chatbot.chat_history_ids = None # Clear history on error


if __name__ == "__main__":
    main()
