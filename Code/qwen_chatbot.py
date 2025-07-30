# qwen_chatbot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        # Load tokenizer and model on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}  # Forces CPU usage
        )
        self.history = []

    def generate_response(self, user_input: str) -> str:
        messages = self.history + [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        generated_ids = self.model.generate(**inputs, max_new_tokens=1000)
        response_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response
