print("running...")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen1.5-4B-Chat"
save_path = "models/qwen"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("Qwen model and tokenizer saved to:", save_path)
