import os
from flask import Flask, render_template, request
from inference import unified_predict
from transformers import AutoTokenizer, pipeline
import torch
from transformers import AutoModelForCausalLM
import ngrok
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ngrok setup
ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "30hEmS9aSOQbYq4QvNqUBBt6jIi_4Cfofec6wU2zvZdmKtwDe")
if not ngrok_token:
    raise ValueError("NGROK_AUTH_TOKEN not set in .env")

ngrok.set_auth_token(ngrok_token)
listener = ngrok.forward("127.0.0.1:5000", domain="safe-polliwog-manually.ngrok-free.app")
print(f"Public URL: {listener.url}")

# Load LLaMA model
print("Loading LLaMA model...")
model_path = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# LLM interaction function
def generate_dermatologist_advice(predicted_disease):
    prompt = f"""
You are a highly knowledgeable and empathetic dermatologist AI assistant. Your role is to help users understand their skin-related symptoms through interactive questioning and provide general suggestions, possible causes, lifestyle recommendations, and advice on whether they should consult a medical professional.

Behavioral Rules:

Always start by acknowledging the user's input and showing empathy.

Ask relevant follow-up questions to narrow down the issue. Examples include:
- How long have you had this symptom?
- What does the affected area look like?
- Does it itch, burn, or cause pain?
- Do you have any allergies or medical conditions?
- How many hours do you sleep per night?
- How much water do you drink daily?
- Have you used any skincare products recently?

Based on user responses, provide:
- A list of possible skin conditions (e.g., eczema, acne, psoriasis).
- General treatments or remedies (e.g., moisturizers, avoiding irritants, OTC creams).
- Lifestyle recommendations (hydration, sleep, diet).
- Clear warning signs that indicate the need to consult a real dermatologist.

Always mention that this is not a medical diagnosis and recommend visiting a healthcare provider for confirmation.

Tone & Style:
- Friendly, conversational, clear
- Avoid technical jargon unless the user is medically trained
- Use bullet points for clarity when listing treatments or suggestions

Here is the input:

[DISEASE: {predicted_disease}]

Now, provide:
- Diagnosis Summary
- Questions for the User
- Suggestions / Treatment Plan
- Notes on when to see a dermatologist
"""
    response = llama_pipeline(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    llm_response = ""

    if request.method == "POST":
        symptom_text = request.form.get("symptoms")
        image_file = request.files.get("image")

        image_path = None
        if image_file and image_file.filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

        # Unified prediction (either symptom or image)
        prediction = unified_predict(input_text=symptom_text, image_path=image_path)

        # Get LLM-generated explanation/suggestion
        llm_response = generate_dermatologist_advice(prediction)

    return render_template("index.html", prediction=prediction, advice=llm_response)

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=5000)
