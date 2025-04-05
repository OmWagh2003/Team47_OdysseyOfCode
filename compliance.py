from huggingface_hub import InferenceClient

# Replace with your Hugging Face token
HF_TOKEN = "your_api_key_here"

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=HF_TOKEN
)

response = client.text_generation(
    prompt="What are the eligibility requirements for a staffing company in a government RFP?",
    max_new_tokens=300,
    temperature=0.7
)

print(response)
